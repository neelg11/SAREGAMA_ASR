#!/usr/bin/env python3
"""
Generic TEST-split inference for any finetuned Whisper LoRA adapter.
Adapters kept LIVE (no merge) — generate on the inner Whisper, matching the
training eval path.

Works for any run by overriding --model_id / --adapter / --language:
  • medium encoder-only : --model_id openai/whisper-medium
                          --adapter  outputs/whisper-medium-enc-lora
                          --language en
  • large-v3-turbo encdec: --model_id openai/whisper-large-v3-turbo
                          --adapter  outputs/whisper-large-v3-turbo-lora
                          --language hi

Decode strategies (--decode_strategy): greedy | beam | beam_reppen

Saves under: predictions/<model_short>_<strategy>/
    test_predictions_<ts>.jsonl  /  .txt  /  test_summary_<ts>.json  /  infer_<ts>.log

Usage:
    python infer_generic.py                                  # medium enc-only defaults
    python infer_generic.py --decode_strategy beam --limit 500
    python infer_generic.py --model_id openai/whisper-large-v3-turbo \
                            --adapter outputs/whisper-large-v3-turbo-lora --language hi
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import librosa
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ── DEFAULTS: medium encoder-only run ──
DEF_MODEL_ID  = "openai/whisper-medium"
DEF_ADAPTER   = "outputs/whisper-medium-enc-lora"
DEF_LANGUAGE  = "hi"

TEST_MANIFEST = "data/training_data_final/manifests/test.jsonl"
OUT_ROOT      = "predictions"
TASK          = "transcribe"
SAMPLE_RATE   = 16_000
MAX_LENGTH    = 448

DECODE_STRATEGIES = {
    "greedy":      {"num_beams": 1},
    "beam":        {"num_beams": 5},
    "beam_reppen": {"num_beams": 5, "repetition_penalty": 1.1},
}

logger = logging.getLogger("infer_generic")


def model_short_name(model_id): return model_id.split("/")[-1]


def setup_logging(out_dir, ts):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"infer_{ts}.log")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return log_path


def fmt_hms(seconds):
    h, rem = divmod(int(seconds), 3600); m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_audio(path):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32), len(audio) / SAMPLE_RATE
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return np.zeros(SAMPLE_RATE, dtype=np.float32), 0.0


def build_model(model_id, adapter_dir, language, device):
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=TASK)
    base = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens    = []

    peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    peft_model.eval(); peft_model.to(device)

    whisper_core = peft_model.base_model.model
    whisper_core.eval()
    return processor, whisper_core


@torch.inference_mode()
def transcribe_batch(audio_list, processor, model, language, device, gen_kwargs):
    inputs = processor.feature_extractor(audio_list, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    feats = inputs.input_features.to(device, dtype=next(model.parameters()).dtype)
    gen_ids = model.generate(
        feats, language=language, task=TASK, max_length=MAX_LENGTH, **gen_kwargs
    )
    return [t.strip() for t in processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",        default=DEF_MODEL_ID)
    p.add_argument("--adapter",         default=DEF_ADAPTER)
    p.add_argument("--language",        default=DEF_LANGUAGE)
    p.add_argument("--manifest",        default=TEST_MANIFEST)
    p.add_argument("--out_root",        default=OUT_ROOT)
    p.add_argument("--decode_strategy", default="greedy", choices=list(DECODE_STRATEGIES.keys()))
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--limit",           type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_kwargs = DECODE_STRATEGIES[args.decode_strategy]

    mshort  = model_short_name(args.model_id)
    out_dir = os.path.join(args.out_root, f"{mshort}_{args.decode_strategy}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(out_dir, ts)

    logger.info(f"Device   : {device}")
    logger.info(f"Model    : {args.model_id}")
    logger.info(f"Adapter  : {args.adapter}  (LIVE, not merged)")
    logger.info(f"Language : {args.language}")
    logger.info(f"Strategy : {args.decode_strategy}  ->  {gen_kwargs}")
    logger.info(f"Out dir  : {out_dir}")

    t_load0 = time.time()
    logger.info("Loading model …")
    processor, model = build_model(args.model_id, args.adapter, args.language, device)
    load_time = time.time() - t_load0
    logger.info(f"Model loaded in {load_time:.1f}s")

    rows = load_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info(f"Test samples: {len(rows)}")

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    results, preds_all, refs_all = [], [], []
    total_audio_sec, total_infer_sec = 0.0, 0.0

    t_run0 = time.time()
    for i in tqdm(range(0, len(rows), args.batch_size), desc="Transcribing"):
        batch_rows = rows[i : i + args.batch_size]
        audio_list, durations = [], []
        for r in batch_rows:
            a, dur = load_audio(r["audio"]); audio_list.append(a); durations.append(dur)
        total_audio_sec += sum(durations)

        t_inf0 = time.time()
        preds = transcribe_batch(audio_list, processor, model, args.language, device, gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_sec += time.time() - t_inf0

        for r, pred in zip(batch_rows, preds):
            ref = r["text"].strip()
            try:
                s_cer = cer_metric.compute(predictions=[pred], references=[ref])
                s_wer = wer_metric.compute(predictions=[pred], references=[ref])
            except Exception:
                s_cer, s_wer = None, None
            results.append({
                "audio": r["audio"], "reference": ref, "prediction": pred,
                "cer": round(s_cer, 4) if s_cer is not None else None,
                "wer": round(s_wer, 4) if s_wer is not None else None,
            })
            preds_all.append(pred); refs_all.append(ref)

    wall_time = time.time() - t_run0
    corpus_cer = cer_metric.compute(predictions=preds_all, references=refs_all)
    corpus_wer = wer_metric.compute(predictions=preds_all, references=refs_all)

    n = len(results)
    sec_per_sample = wall_time / n if n else 0.0
    rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else None

    pred_path = os.path.join(out_dir, f"test_predictions_{ts}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    txt_path = os.path.join(out_dir, f"test_predictions_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(results):
            f.write(f"[{idx}] {os.path.basename(row['audio'])}\n")
            f.write(f"  REF : {row['reference']}\n")
            f.write(f"  HYP : {row['prediction']}\n")
            f.write(f"  CER : {row['cer']}   WER : {row['wer']}\n\n")

    summary = {
        "timestamp": ts, "model": args.model_id, "adapter": args.adapter,
        "language": args.language, "merged": False, "manifest": args.manifest,
        "decode_strategy": args.decode_strategy, "gen_kwargs": gen_kwargs,
        "num_samples": n, "batch_size": args.batch_size,
        "corpus_cer": round(corpus_cer, 4), "corpus_wer": round(corpus_wer, 4),
        "timing": {
            "model_load_sec":   round(load_time, 2),
            "total_wall_sec":   round(wall_time, 2),
            "total_wall_hms":   fmt_hms(wall_time),
            "pure_infer_sec":   round(total_infer_sec, 2),
            "total_audio_sec":  round(total_audio_sec, 2),
            "sec_per_sample":   round(sec_per_sample, 3),
            "samples_per_sec":  round(n / wall_time, 2) if wall_time else None,
            "real_time_factor": round(rtf, 3) if rtf is not None else None,
        },
    }
    summary_path = os.path.join(out_dir, f"test_summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 55)
    logger.info(f"Model            : {mshort}")
    logger.info(f"Strategy         : {args.decode_strategy}  {gen_kwargs}")
    logger.info(f"Corpus CER       : {corpus_cer:.4f}")
    logger.info(f"Corpus WER       : {corpus_wer:.4f}")
    logger.info(f"Samples          : {n}")
    logger.info(f"Model load        : {load_time:.1f}s")
    logger.info(f"Total wall time   : {fmt_hms(wall_time)} ({wall_time:.1f}s)")
    logger.info(f"Pure inference    : {total_infer_sec:.1f}s")
    if wall_time:
        logger.info(f"Throughput        : {n / wall_time:.2f} samples/sec")
    if rtf is not None:
        logger.info(f"Real-time factor  : {rtf:.3f}  (audio={total_audio_sec:.0f}s)")
    logger.info("-" * 55)
    logger.info(f"Predictions : {pred_path}")
    logger.info(f"Readable    : {txt_path}")
    logger.info(f"Summary     : {summary_path}")
    logger.info(f"Log         : {log_path}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()