# Hindi Singing ASR

A production web inference app for a fine-tuned **Whisper Large-v3 Turbo** model
specialised in **Hindi song-lyric transcription**. The model is loaded lazily on
the first request and automatically unloaded after a period of inactivity, so the
GPU only holds VRAM while someone is actually transcribing — ideal for parking a
demo on an always-on H100 box.

The model always decodes to **Devanagari**; the result is then transliterated
on the fly into any of **ten Indic scripts** (Devanagari, Bengali, Gurmukhi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) using a positional Unicode
mapping.

---

## Highlights

- **Lazy GPU usage** — model is *not* loaded at startup; the first request loads it.
- **Auto-sleep** — a background watchdog unloads the model after `IDLE_TIMEOUT_SECONDS`
  (default 300s): moves it off GPU, deletes objects, `gc.collect()`, `torch.cuda.empty_cache()`.
- **Thread-safe** — all lifecycle transitions guarded by a re-entrant lock.
- **Live monitor** — GPU memory, load state (🟢/🔴), last request, total requests, auto-refreshing.
- **bfloat16 + `torch.inference_mode()`**, processor cached across requests.
- **No build step** — the frontend is a single hand-written HTML/CSS/JS page.

---

## Project layout

```
.
├── app.py            # FastAPI server + ModelManager + watchdog
├── transliterate.py  # Devanagari → 10 scripts positional mapping
├── static/
│   └── index.html    # single-page premium UI
├── requirements.txt
├── Dockerfile
├── launch.sh
└── README.md
```

---

## Run locally

### 1. Install dependencies

Install the CUDA build of torch that matches your driver, then the rest:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Point at your merged checkpoint

The app expects a merged (LoRA already folded in) Whisper checkpoint. By default
it looks for `./whisper-large-v3-turbo-merged`. Override with `MODEL_PATH`:

```bash
export MODEL_PATH=/data/whisper-large-v3-turbo-merged
```

### 3. Start

```bash
python app.py
# or
./launch.sh
```

Open **http://localhost:7860**.

---

## Run with Docker

The image uses a CUDA runtime base and does **not** bake in the weights — mount
them at run time.

```bash
# Build
docker build -t hindi-singing-asr .

# Run (mount the checkpoint, expose the port, request GPUs)
docker run --rm -it --gpus all \
  -p 7860:7860 \
  -v /data/whisper-large-v3-turbo-merged:/models/whisper-large-v3-turbo-merged:ro \
  hindi-singing-asr
```

Then open **http://localhost:7860**.

---

## Configuration

All settings are environment variables:

| Variable                   | Default                          | Description                                  |
| -------------------------- | -------------------------------- | -------------------------------------------- |
| `MODEL_PATH`               | `whisper-large-v3-turbo-merged`  | Path to the merged checkpoint                |
| `IDLE_TIMEOUT_SECONDS`     | `300`                            | Idle time before the model is unloaded       |
| `WATCHDOG_INTERVAL_SECONDS`| `30`                             | How often the watchdog checks for idleness   |
| `HOST`                     | `0.0.0.0`                        | Bind host                                    |
| `PORT`                     | `7860`                           | Bind port                                    |

---

## API

| Method | Path              | Purpose                                            |
| ------ | ----------------- | -------------------------------------------------- |
| `GET`  | `/`               | Serves the web UI                                  |
| `GET`  | `/api/scripts`    | List of supported output scripts                   |
| `GET`  | `/api/status`     | Model/GPU status for the monitor panel             |
| `POST` | `/api/transcribe` | Transcribe audio → Devanagari + transliterated text |

`POST /api/transcribe` (multipart form):

- `audio` — the audio file (wav/mp3/flac/m4a/webm)
- `beam_size` — int, default `5`
- `temperature` — float, default `0.0` (0 = deterministic beam search)
- `language` — default `hi`
- `target_script` — one of the script ids, default `devanagari`

Response:

```json
{
  "devanagari": "…",
  "transliterated": "…",
  "target_script": "telugu",
  "stats": {
    "audio_duration_s": 12.4,
    "inference_time_s": 1.8,
    "real_time_factor": 0.145,
    "gpu_name": "NVIDIA H100 PCIe",
    "model_was_sleeping": true
  }
}
```

---

## Notes on transliteration

Transliteration is a **positional Unicode mapping**, faithful to the source
registry: each Devanagari code point maps to the same slot in the target script.
It is intentionally simple and not phonetically perfect — scripts with structural
differences (e.g. Tamil's reduced consonant set, missing matras in some scripts)
will be approximate. Script-specific correction rules can be layered on later.
The same table is mirrored client-side, so switching the output script after a
transcription re-renders instantly without re-running the model.
