"""
app.py
======

Production web inference server for a fine-tuned Whisper Large-v3 Turbo model
specialised for Hindi song-lyric transcription.

Design highlights
-----------------
* **Lazy GPU usage** — the model is *not* loaded at startup. The first
  transcription request loads it; subsequent requests reuse it.
* **Automatic sleep** — a background watchdog unloads the model after a
  configurable idle period (default 5 minutes), moving it off the GPU,
  deleting the objects, running ``gc.collect()`` and
  ``torch.cuda.empty_cache()`` so VRAM is fully released.
* **Thread-safe** — all model lifecycle transitions are guarded by a single
  re-entrant lock so concurrent requests cannot race the loader/unloader.
* **Transliteration** — Whisper returns Devanagari; the result is mapped into
  any of ten Indic scripts via :mod:`transliterate`.

Run locally::

    python app.py

Then open http://localhost:7860
"""

from __future__ import annotations

import gc
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transliterate import list_scripts, transliterate

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.environ.get("MODEL_PATH", "whisper-large-v3-turbo-merged")
IDLE_TIMEOUT_SECONDS: int = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "300"))
WATCHDOG_INTERVAL_SECONDS: int = int(os.environ.get("WATCHDOG_INTERVAL_SECONDS", "30"))
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "7860"))
TARGET_SAMPLE_RATE: int = 16_000  # Whisper expects 16 kHz mono.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whisper-asr")


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    """Lightweight, serialisable view of the manager's runtime state."""

    loaded: bool = False
    last_request_time: Optional[float] = None
    total_requests: int = 0
    device: str = "cpu"
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0


class ModelManager:
    """Thread-safe singleton that owns the Whisper model lifecycle.

    The processor is loaded once and kept resident (it is CPU-side and cheap);
    only the model weights are moved/deleted to free VRAM.
    """

    _instance: Optional["ModelManager"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        # Double-checked locking for a process-wide singleton.
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        self._lock = threading.RLock()
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

        self._loaded: bool = False
        self._last_used: Optional[float] = None
        self._total_requests: int = 0

        self._use_cuda: bool = torch.cuda.is_available()
        self._device: str = "cuda" if self._use_cuda else "cpu"
        self._dtype: torch.dtype = torch.bfloat16 if self._use_cuda else torch.float32

        logger.info(
            "ModelManager initialised | device=%s dtype=%s model_path=%s",
            self._device, self._dtype, MODEL_PATH,
        )

    # -- lifecycle ----------------------------------------------------------

    def load_model(self) -> None:
        """Load processor (once) and model weights onto the target device.

        Idempotent: returns immediately if the model is already resident.
        """
        with self._lock:
            if self._loaded:
                return

            # Imported lazily so importing this module is cheap and so a missing
            # transformers install fails only when a request actually arrives.
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )

            t0 = time.perf_counter()
            logger.info("Loading model from %s ...", MODEL_PATH)

            if self._processor is None:
                # Processor is loaded once and reused for the process lifetime.
                self._processor = AutoProcessor.from_pretrained(MODEL_PATH)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_PATH,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self._device)
            model.eval()

            self._model = model
            self._loaded = True

            logger.info(
                "Model loaded in %.2fs on %s", time.perf_counter() - t0, self._device
            )

    def unload_model(self) -> None:
        """Move the model off-GPU, delete it, and reclaim VRAM."""
        with self._lock:
            if not self._loaded:
                return

            logger.info("Unloading model (idle timeout) ...")
            try:
                if self._model is not None:
                    # Move to CPU first so CUDA tensors are released cleanly.
                    self._model.to("cpu")
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error moving model to CPU during unload")

            self._model = None
            self._loaded = False

            gc.collect()
            if self._use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Model unloaded; VRAM released")

    # -- inference ----------------------------------------------------------

    def transcribe(
        self,
        audio: torch.Tensor,
        *,
        beam_size: int = 5,
        temperature: float = 0.0,
        language: str = "hi",
    ) -> str:
        """Run transcription on a 16 kHz mono waveform tensor.

        The model is loaded on demand if it is currently sleeping. Returns the
        raw Devanagari transcript produced by Whisper.
        """
        with self._lock:
            if not self._loaded:
                self.load_model()

            assert self._model is not None and self._processor is not None

            inputs = self._processor(
                audio.numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self._device, dtype=self._dtype)

            # temperature == 0 → deterministic beam search; otherwise sample.
            gen_kwargs: Dict[str, Any] = {
                "num_beams": max(1, int(beam_size)),
                "language": language,
                "task": "transcribe",
            }
            if temperature and temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)

            with torch.inference_mode():
                generated_ids = self._model.generate(input_features, **gen_kwargs)

            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            self._last_used = time.time()
            self._total_requests += 1
            return text

    # -- introspection ------------------------------------------------------

    def maybe_unload(self, idle_timeout: int) -> None:
        """Unload the model if it has been idle longer than ``idle_timeout``."""
        with self._lock:
            if not self._loaded or self._last_used is None:
                return
            if time.time() - self._last_used >= idle_timeout:
                self.unload_model()

    def stats(self) -> ModelStats:
        """Return a snapshot of current runtime state (cheap, lock-guarded)."""
        with self._lock:
            used_mb = total_mb = 0.0
            gpu_name = None
            if self._use_cuda:
                gpu_name = torch.cuda.get_device_name(0)
                used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

            return ModelStats(
                loaded=self._loaded,
                last_request_time=self._last_used,
                total_requests=self._total_requests,
                device=self._device,
                gpu_name=gpu_name,
                gpu_memory_used_mb=round(used_mb, 1),
                gpu_memory_total_mb=round(total_mb, 1),
            )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(raw: bytes) -> tuple[torch.Tensor, float]:
    """Decode arbitrary audio bytes into a 16 kHz mono float32 tensor.

    Returns ``(waveform, duration_seconds)``. Raises ``ValueError`` if the
    bytes cannot be decoded.
    """
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the API
        raise ValueError(f"Could not decode audio: {exc}") from exc

    # Down-mix to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed.
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )

    waveform = waveform.squeeze(0).to(torch.float32)
    duration = waveform.shape[-1] / TARGET_SAMPLE_RATE
    return waveform, duration


# ---------------------------------------------------------------------------
# Watchdog thread
# ---------------------------------------------------------------------------

class InactivityWatchdog(threading.Thread):
    """Daemon thread that periodically unloads an idle model."""

    def __init__(self, manager: ModelManager) -> None:
        super().__init__(daemon=True, name="inactivity-watchdog")
        self._manager = manager
        self._stop_event = threading.Event()

    def run(self) -> None:  # noqa: D102
        logger.info(
            "Watchdog started (idle_timeout=%ds, interval=%ds)",
            IDLE_TIMEOUT_SECONDS, WATCHDOG_INTERVAL_SECONDS,
        )
        while not self._stop_event.is_set():
            self._stop_event.wait(WATCHDOG_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            try:
                self._manager.maybe_unload(IDLE_TIMEOUT_SECONDS)
            except Exception:  # pragma: no cover - never let the thread die
                logger.exception("Watchdog tick failed")

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

manager = ModelManager()
_watchdog: Optional[InactivityWatchdog] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the watchdog on boot, stop it on shutdown. No model load here."""
    global _watchdog
    _watchdog = InactivityWatchdog(manager)
    _watchdog.start()
    logger.info("Server ready — model will load on first request")
    try:
        yield
    finally:
        if _watchdog is not None:
            _watchdog.stop()
        manager.unload_model()


app = FastAPI(title="Hindi Singing ASR", version="1.0.0", lifespan=lifespan)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/scripts")
def get_scripts() -> JSONResponse:
    """Return the list of supported target scripts for the UI."""
    return JSONResponse({"scripts": list_scripts()})


@app.get("/api/status")
def get_status() -> JSONResponse:
    """Return live model/GPU status for the monitoring panel."""
    s = manager.stats()
    return JSONResponse(
        {
            "loaded": s.loaded,
            "status_label": "Model Loaded" if s.loaded else "Model Sleeping",
            "device": s.device,
            "gpu_name": s.gpu_name,
            "gpu_memory_used_mb": s.gpu_memory_used_mb,
            "gpu_memory_total_mb": s.gpu_memory_total_mb,
            "last_request_time": s.last_request_time,
            "total_requests": s.total_requests,
            "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS,
        }
    )


@app.post("/api/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
    language: str = Form("hi"),
    target_script: str = Form("devanagari"),
) -> JSONResponse:
    """Transcribe an uploaded audio file and transliterate the result.

    Returns the raw Devanagari transcript, the transliterated transcript in the
    requested target script, and timing statistics.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        waveform, duration = load_audio(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_was_sleeping = not manager.stats().loaded

    t0 = time.perf_counter()
    try:
        devanagari = manager.transcribe(
            waveform,
            beam_size=beam_size,
            temperature=temperature,
            language=language,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    inference_time = time.perf_counter() - t0

    transliterated = transliterate(devanagari, target_script)
    rtf = inference_time / duration if duration > 0 else 0.0
    stats = manager.stats()

    return JSONResponse(
        {
            "devanagari": devanagari,
            "transliterated": transliterated,
            "target_script": target_script,
            "stats": {
                "audio_duration_s": round(duration, 2),
                "inference_time_s": round(inference_time, 2),
                "real_time_factor": round(rtf, 3),
                "gpu_name": stats.gpu_name or stats.device.upper(),
                "model_was_sleeping": model_was_sleeping,
            },
        }
    )


# Mount static assets (JS/CSS) after routes so "/" stays handled above.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
