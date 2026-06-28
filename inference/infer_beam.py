#!/usr/bin/env python3

import os
import json
import time
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import librosa
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

DEF_MODEL_ID  = "openai/whisper-large-v3-turbo"
DEF_ADAPTER   = "outputs/whisper-large-v3-turbo-lora"
DEF_LANGUAGE  = "hi"

TEST_MANIFEST = "data/training_data_final/manifests/test.jsonl"
OUT_ROOT      = "predictions"
TASK          = "transcribe"
SAMPLE_RATE   = 16_000
MAX_LENGTH    = 448

AUDIO_WORKERS = 4
_AUDIO_POOL   = ThreadPoolExecutor(max_workers=AUDIO_WORKERS)

logger = logging.getLogger("infer_beam")


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
def transcribe_nbest(audio_list, processor, model, language, device,
                     num_beams, num_return_sequences):
    """
    PLAIN BEAM. Returns list (len=batch) of lists (len=num_return_sequences) of
    dicts, best-first per sample (beam already returns them sorted).
    """
    inputs = processor.feature_extractor(audio_list, sampling_rate=SAMPLE_RATE,
                                         return_tensors="pt")
    feats = inputs.input_features.to(device, dtype=next(model.parameters()).dtype)

    out = model.generate(
        feats,
        language=language,
        task=TASK,
        max_length=MAX_LENGTH,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        length_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True,
        # NO do_sample -> plain beam -> sequences_scores populated, fast, no OOM
    )

    assert out.sequences_scores is not None, \
        "sequences_scores is None — not in beam mode? check generate() args"

    seqs   = out.sequences            # (batch * num_return_sequences, seq_len)
    scores = out.sequences_scores     # (batch * num_return_sequences,)

    bsz = len(audio_list)
    results = []
    for b in range(bsz):
        sample_hyps = []
        for k in range(num_return_sequences):
            idx  = b * num_return_sequences + k
            ids  = seqs[idx]
            text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            n_tok = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
            n_tok = max(n_tok, 1)
            norm_lp = float(scores[idx].item())
            raw_lp  = norm_lp * (n_tok ** 1.0)
            sample_hyps.append({
                "text": text,
                "am_logprob_norm": round(norm_lp, 6),
                "am_logprob": round(raw_lp, 6),
                "token_count": n_tok,
                "token_ids": ids.tolist(),
            })
        results.append(sample_hyps)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",             default=DEF_MODEL_ID)
    p.add_argument("--adapter",              default=DEF_ADAPTER)
    p.add_argument("--language",             default=DEF_LANGUAGE)
    p.add_argument("--manifest",             default=TEST_MANIFEST)
    p.add_argument("--out_root",             default=OUT_ROOT)
    p.add_argument("--num_beams",            type=int, default=5)
    p.add_argument("--num_return_sequences", type=int, default=5)
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--limit",                type=int, default=0)
    p.add_argument("--save_token_ids",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.num_return_sequences <= args.num_beams, \
        "num_return_sequences must be <= num_beams"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mshort  = model_short_name(args.model_id)
    out_dir = os.path.join(args.out_root, f"{mshort}_nbest_b{args.num_beams}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(out_dir, ts)

    logger.info(f"Device       : {device}")
    logger.info(f"Model        : {args.model_id}")
    logger.info(f"Adapter      : {args.adapter}  (LIVE, not merged)")
    logger.info(f"Language     : {args.language}")
    logger.info(f"Decode       : PLAIN BEAM  beams={args.num_beams}  return={args.num_return_sequences}")
    logger.info(f"Out dir      : {out_dir}")

    t0 = time.time()
    processor, model = build_model(args.model_id, args.adapter, args.language, device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    rows = load_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info(f"Test samples : {len(rows)}")

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    nbest_path = os.path.join(out_dir, f"nbest_{ts}.jsonl")
    nbest_f = open(nbest_path, "w", encoding="utf-8")

    top1_preds, refs_all = [], []
    total_audio_sec, total_infer_sec = 0.0, 0.0
    t_run0 = time.time()

    for i in tqdm(range(0, len(rows), args.batch_size), desc="N-best"):
        batch_rows = rows[i : i + args.batch_size]

        loaded = list(_AUDIO_POOL.map(load_audio, [r["audio"] for r in batch_rows]))
        audio_list = [a for a, _ in loaded]
        durations  = [d for _, d in loaded]
        total_audio_sec += sum(durations)

        t_inf0 = time.time()
        batch_nbest = transcribe_nbest(
            audio_list, processor, model, args.language, device,
            args.num_beams, args.num_return_sequences,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_sec += time.time() - t_inf0

        for r, hyps in zip(batch_rows, batch_nbest):
            ref = r["text"].strip()
            if not args.save_token_ids:
                for h in hyps:
                    h.pop("token_ids", None)
            for rank, h in enumerate(hyps):
                h["rank"] = rank
            record = {"audio": r["audio"], "reference": ref, "hypotheses": hyps}
            nbest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            top1_preds.append(hyps[0]["text"])
            refs_all.append(ref)

    nbest_f.close()
    wall_time = time.time() - t_run0

    corpus_cer = cer_metric.compute(predictions=top1_preds, references=refs_all)
    corpus_wer = wer_metric.compute(predictions=top1_preds, references=refs_all)

    n   = len(top1_preds)
    rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else None

    summary = {
        "timestamp": ts, "model": args.model_id, "adapter": args.adapter,
        "language": args.language, "merged": False, "manifest": args.manifest,
        "decode": "plain_beam",
        "num_beams": args.num_beams, "num_return_sequences": args.num_return_sequences,
        "length_penalty": 1.0,
        "num_samples": n, "batch_size": args.batch_size,
        "top1_corpus_cer": round(corpus_cer, 4),
        "top1_corpus_wer": round(corpus_wer, 4),
        "nbest_file": nbest_path,
        "score_fields": {
            "am_logprob_norm": "length-normalized sequence log-prob (fuse this)",
            "am_logprob": "raw summed sequence log-prob",
        },
        "timing": {
            "total_wall_sec":   round(wall_time, 2),
            "total_wall_hms":   fmt_hms(wall_time),
            "pure_infer_sec":   round(total_infer_sec, 2),
            "total_audio_sec":  round(total_audio_sec, 2),
            "samples_per_sec":  round(n / wall_time, 2) if wall_time else None,
            "real_time_factor": round(rtf, 3) if rtf is not None else None,
        },
    }
    summary_path = os.path.join(out_dir, f"nbest_summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 55)
    logger.info(f"Top-1 corpus CER : {corpus_cer:.4f}")
    logger.info(f"Top-1 corpus WER : {corpus_wer:.4f}")
    logger.info(f"Samples          : {n}")
    logger.info(f"Wall time        : {fmt_hms(wall_time)}")
    if rtf is not None:
        logger.info(f"Real-time factor : {rtf:.3f}")
    logger.info("-" * 55)
    logger.info(f"N-best   : {nbest_path}")
    logger.info(f"Summary  : {summary_path}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()