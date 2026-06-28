#!/usr/bin/env python3
"""
Re-evaluate an existing predictions JSONL — no model, no audio, no inference.
Computes corpus CER/WER against:
  1. RAW reference            (the manifest text already stored in the file)
  2. ROUNDTRIP reference      (decode(encode(text)) truncated to 448 tokens —
                               exactly what the training compute_metrics used)

Reads the JSONL produced by infer_test_match_train.py, which has fields:
    {"audio", "reference", "prediction", "cer", "wer", ...}

Usage:
    python eval_predictions.py --preds predictions/whisper-large-v3-turbo_greedy/test_predictions_XXXX.jsonl
    python eval_predictions.py --preds <file>.jsonl --show_mismatches 20
"""

import os
import json
import argparse

import evaluate
from transformers import WhisperTokenizer

MODEL_ID         = "openai/whisper-large-v3-turbo"
LANGUAGE         = "hi"
TASK             = "transcribe"
MAX_LABEL_LENGTH = 448


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def roundtrip(tokenizer, text):
    ids = tokenizer(text, max_length=MAX_LABEL_LENGTH, truncation=True).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="Path to predictions JSONL")
    p.add_argument("--show_mismatches", type=int, default=0,
                   help="Print N rows where raw != roundtrip reference")
    p.add_argument("--out", default=None,
                   help="Optional path to write a re-eval summary JSON")
    return p.parse_args()


def main():
    args = parse_args()

    rows = load_jsonl(args.preds)
    n = len(rows)
    print(f"Loaded {n} predictions from {args.preds}")

    # Load tokenizer only (no model weights, fast)
    print("Loading tokenizer …")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    preds      = [r["prediction"].strip() for r in rows]
    refs_raw   = [r["reference"].strip()  for r in rows]
    refs_rt    = [roundtrip(tokenizer, r["reference"]) for r in rows]

    n_diff = sum(1 for a, b in zip(refs_raw, refs_rt) if a != b)

    # ── Corpus metrics ──
    cer_raw = cer_metric.compute(predictions=preds, references=refs_raw)
    wer_raw = wer_metric.compute(predictions=preds, references=refs_raw)
    cer_rt  = cer_metric.compute(predictions=preds, references=refs_rt)
    wer_rt  = wer_metric.compute(predictions=preds, references=refs_rt)

    print("=" * 60)
    print(f"Samples                      : {n}")
    print(f"Rows where raw != roundtrip  : {n_diff}/{n} ({100*n_diff/n:.1f}%)")
    print("-" * 60)
    print(f"[RAW reference]   CER : {cer_raw:.4f}   WER : {wer_raw:.4f}")
    print(f"[ROUNDTRIP ref]   CER : {cer_rt:.4f}   WER : {wer_rt:.4f}   (matches training)")
    print("=" * 60)

    # ── Optionally show where references differ ──
    if args.show_mismatches > 0:
        shown = 0
        print(f"\nFirst {args.show_mismatches} reference mismatches:\n")
        for r, raw, rt in zip(rows, refs_raw, refs_rt):
            if raw != rt:
                print(f"  AUDIO   : {os.path.basename(r['audio'])}")
                print(f"  RAW     : {raw}")
                print(f"  RTRIP   : {rt}")
                print(f"  HYP     : {r['prediction'].strip()}")
                print()
                shown += 1
                if shown >= args.show_mismatches:
                    break
        if shown == 0:
            print("  (none — raw and roundtrip references are identical on every row)")

    # ── Optional summary file ──
    if args.out:
        summary = {
            "preds_file":      args.preds,
            "num_samples":     n,
            "rows_differing":  n_diff,
            "pct_differing":   round(100 * n_diff / n, 2) if n else 0.0,
            "reference_raw":       {"cer": round(cer_raw, 4), "wer": round(wer_raw, 4)},
            "reference_roundtrip": {"cer": round(cer_rt, 4),  "wer": round(wer_rt, 4)},
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()