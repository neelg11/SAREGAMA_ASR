#!/usr/bin/env python3
"""
Whisper Medium — Encoder+Decoder LoRA finetuning for Singing ASR (Romanized Hindi).
Launch: torchrun --nproc_per_node=2 train_whisper_medium_encdec_lora.py

Identical to the encoder-only variant in data, system, env, and all
PEFT/transformers-5.x fixes.  ONLY the LoRA scope differs: adapters are now
placed on BOTH encoder and decoder attention (decoder self-attn AND
cross-attn), not the encoder alone.

Fixes vs original:
  FIX-1  patch_peft_whisper_forward()
         PeftModelForSeq2SeqLM.forward always injects input_ids=None.  In
         transformers-5.x both WhisperForConditionalGeneration.forward and
         WhisperModel.forward accept **kwargs and propagate them, so that None
         arrives at self.decoder(input_ids=decoder_input_ids, **kwargs) twice:
             TypeError: got multiple values for keyword argument 'input_ids'
         For LoRA (no virtual tokens) PEFT's forward is a pure passthrough; we
         replace it — and generate() — with Whisper-aware shims.

  FIX-2  processing_class=processor (not processor.feature_extractor)
         Trainer sets self.tokenizer = processing_class for backwards compat.
         Passing a bare FeatureExtractor means self.tokenizer has no
         .pad_token_id, breaking _pad_tensors_to_max_len on the first eval.

  FIX-3  Tuple guard in compute_metrics
         pred.predictions can be (token_ids, scores) when beams > 1 or
         return_dict_in_generate=True.

  FIX-4  ddp_find_unused_parameters=False
         Every trainable (LoRA) parameter — now in BOTH encoder and decoder —
         lies on the path to the loss, so there are no unused parameters and
         False is correct.  It is also faster than True, which additionally
         interacts badly with gradient checkpointing.

  FIX-5  Checkpoint resume sorted by step number, not ctime
         ctime is unreliable on NFS/remote filesystems.

  FIX-6  num_proc / batch_size capped to available CPUs
         Hardcoded 64 workers / 512 batch_size crashes on smaller nodes.

  FIX-7  getattr guard for generation_num_beams
         The attribute was added to Seq2SeqTrainingArguments in a later
         transformers release; guard with getattr to stay compatible.

  FIX-8  accelerator.unwrap_model for .generate() in prediction_step
         DDP-wrapped models do not expose .generate(); must unwrap first.

  FIX-9  compute_loss_context_manager() wrapped around generate + loss fwd
         During training, compute_loss runs inside the Trainer's
         compute_loss_context_manager() which activates
         torch.amp.autocast(dtype=bfloat16).  prediction_step is called
         outside that context, so float32 input_features collide with
         bfloat16 model weights and raise:
             RuntimeError: Input type (float) and bias type (c10::BFloat16)
                           should be the same
         Fix: wrap both generate() and the loss forward pass explicitly.
"""

import os
import re
import sys
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import evaluate
import wandb
import librosa

from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
# MODEL_ID         = "openai/whisper-medium"
# LANGUAGE         = "hi"
# TASK             = "transcribe"
# MAX_LABEL_LENGTH = 448
# SAMPLING_RATE    = 16_000
# SEED             = 42

# TRAIN_MANIFEST   = "data/training_data_final/manifests/train.jsonl"
# VAL_MANIFEST     = "data/training_data_final/manifests/val.jsonl"
# TEST_MANIFEST    = "data/training_data_final/manifests/test.jsonl"

# # NOTE: distinct output dir so FIX-5 resume logic never picks up the
# # encoder-only run's checkpoints.
# OUTPUT_DIR       = "outputs/whisper-medium-encdec-lora"
# CACHE_DIR        = "cache/hf_datasets"

# WANDB_PROJECT    = "whisper-singing-asr"
# WANDB_RUN_NAME   = "medium-encdec-lora-r32"

MODEL_ID         = "openai/whisper-large-v3-turbo"
LANGUAGE         = "hi"
TASK             = "transcribe"
MAX_LABEL_LENGTH = 448
SAMPLING_RATE    = 16_000
SEED             = 42

TRAIN_MANIFEST   = "data/training_data_final/manifests/train.jsonl"
VAL_MANIFEST     = "data/training_data_final/manifests/val.jsonl"
TEST_MANIFEST    = "data/training_data_final/manifests/test.jsonl"

OUTPUT_DIR       = "outputs/whisper-large-v3-turbo-lora"
CACHE_DIR        = "cache/hf_datasets"

WANDB_PROJECT    = "whisper-singing-asr"
WANDB_RUN_NAME   = "large-v3-turbo-encdec-lora-r32"

# LoRA — encoder AND decoder via regex.
#   • encoder self_attn        : q/k/v/out_proj
#   • decoder self_attn        : q/k/v/out_proj
#   • decoder encoder_attn     : q/k/v/out_proj   (cross-attention)
# To restrict to self-attention only (a stricter mirror of the encoder-only
# run), drop "encoder_attn":
#     r"model\.(encoder|decoder)\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"
LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=r"model\.(encoder|decoder)\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)",
)

TRAINING_ARGS = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=500,
    num_train_epochs=20,
    bf16=True,
    bf16_full_eval=True,   # FIX-9a: bf16=True only covers training; this covers eval too
    fp16=False,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=MAX_LABEL_LENGTH,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    ddp_find_unused_parameters=False,  # all encoder+decoder LoRA params are on the loss path
    report_to=["wandb"],
    run_name=WANDB_RUN_NAME,
    seed=SEED,
    remove_unused_columns=False,
    tf32=True,
)

# ──────────────────────────────────────────────
# Verify LoRA hits BOTH encoder and decoder
# ──────────────────────────────────────────────
# `\.encoder\.` matches the encoder module path but NOT the decoder's
# `encoder_attn` (which appears as `.encoder_attn.`), so the two checks below
# do not alias each other.
_ENCODER_RE = re.compile(r"\.encoder\.")
_DECODER_RE = re.compile(r"\.decoder\.")


def assert_encoder_and_decoder(model) -> None:
    has_enc = has_dec = False
    for name, _ in model.named_parameters():
        if "lora_" in name:
            if _ENCODER_RE.search(name):
                has_enc = True
            if _DECODER_RE.search(name):
                has_dec = True
    if not has_enc:
        raise RuntimeError("No LoRA adapters found on the encoder.")
    if not has_dec:
        raise RuntimeError("No LoRA adapters found on the decoder.")
    logger.info("✓ LoRA adapters verified: present on BOTH encoder and decoder.")


def log_trainable_parameters(model) -> None:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    logger.info(f"Trainable params: {trainable:,} / {total:,}  ({pct:.4f}%)")


# ──────────────────────────────────────────────
# FIX-1: PEFT / Whisper forward compatibility
# ──────────────────────────────────────────────
def patch_peft_whisper_forward(peft_model) -> None:
    """
    PeftModelForSeq2SeqLM.forward always calls:

        self.base_model(input_ids=input_ids, ..., **kwargs)

    with input_ids=None even when the caller never provided it.  In
    transformers-5.x, WhisperForConditionalGeneration.forward and
    WhisperModel.forward both accept **kwargs and pass them downstream, so
    the stale None reaches:

        self.decoder(input_ids=decoder_input_ids, **kwargs)
                                                   ^^^^^^^^ also contains input_ids=None

    raising: TypeError: got multiple values for keyword argument 'input_ids'

    For LoRA there are no virtual tokens; the PEFT forward is a pure
    passthrough.  We replace it — and generate() — with Whisper-aware shims
    as instance attributes.  nn.Module.__call__ looks up instance attributes
    before class methods, so DDP gradient sync (driven by backward hooks
    registered on parameters, not by the forward call itself) is unaffected.
    """
    # WhisperForConditionalGeneration with LoRA layers already embedded
    whisper_core: WhisperForConditionalGeneration = peft_model.base_model.model

    def _forward(input_features=None, **kwargs):
        # Evict any input_ids=None that PEFT would have injected.
        # kwargs here is just {labels: tensor} from our compute_loss, so
        # this pop is a no-op in practice — but it's the critical guard.
        kwargs.pop("input_ids", None)
        return whisper_core(input_features=input_features, **kwargs)

    def _generate(*args, **kwargs):
        # Delegate directly to WhisperForConditionalGeneration.generate(),
        # bypassing PeftModelForSeq2SeqLM's generate (which may similarly
        # mangle kwargs in some PEFT versions).
        return whisper_core.generate(*args, **kwargs)

    peft_model.forward  = _forward   # instance attr — picked up by nn.Module.__call__
    peft_model.generate = _generate  # instance attr — picked up by DDP __getattr__
    logger.info(
        "✓ PeftModel.forward / .generate patched for Whisper "
        "(transformers-5.x + PEFT compatibility)."
    )


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def load_manifests() -> DatasetDict:
    return load_dataset(
        "json",
        data_files={
            "train":      TRAIN_MANIFEST,
            "validation": VAL_MANIFEST,
            "test":       TEST_MANIFEST,
        },
        cache_dir=CACHE_DIR,
    )


def build_preprocess_fn(
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
):
    def preprocess(batch):
        audio_arrays = []
        for path in batch["audio"]:
            try:
                audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
                audio = audio.astype(np.float32)
            except Exception as exc:
                logger.warning(f"Failed to load {path}: {exc}")
                audio = np.zeros(SAMPLING_RATE, dtype=np.float32)
            audio_arrays.append(audio)

        inputs = feature_extractor(
            audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="np"
        )
        batch["input_features"] = inputs.input_features

        labels = tokenizer(
            batch["text"],
            max_length=MAX_LABEL_LENGTH,
            truncation=True,
            padding=False,
        )
        batch["labels"] = labels["input_ids"]
        return batch

    return preprocess


def preprocess_dataset(raw: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
    fn = build_preprocess_fn(processor.feature_extractor, processor.tokenizer)
    # FIX-6: cap to available CPUs; batch_size 64 is safer with parallel audio loading
    num_proc = min(64, os.cpu_count() or 1)
    return raw.map(
        fn,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        remove_columns=["audio", "text"],
        load_from_cache_file=True,
        desc="Preprocessing",
    )


# ──────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────
@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ──────────────────────────────────────────────
# Custom Trainer
# ──────────────────────────────────────────────
class WhisperLoRATrainer(Seq2SeqTrainer):
    """
    compute_loss  — passes input_features + labels explicitly so that the
                    call signature arriving at our patched PEFT forward is
                    unambiguous (no leftover Trainer state in **inputs).

    prediction_step — uses accelerator.unwrap_model() for .generate() because
                      DDP-wrapped models do not expose that method.  The
                      patched _generate() on the PEFT model delegates directly
                      to WhisperForConditionalGeneration.generate().
    """

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        input_features = inputs["input_features"]
        labels         = inputs["labels"]
        # model = DDP(PeftModel); our patched PeftModel.forward handles the rest.
        outputs = model(input_features=input_features, labels=labels)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            # For loss-only eval, super() calls self.compute_loss() which is
            # already overridden above — no generate involved.
            return super().prediction_step(
                model, inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs     = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.args.generation_max_length,
            # FIX-7: generation_num_beams was added in a later transformers release
            "num_beams":  getattr(self.args, "generation_num_beams", None) or 1,
            "language":   LANGUAGE,
            "task":       TASK,
        }

        # FIX-8: DDP models don't expose .generate(); unwrap to the PeftModel
        # whose generate() was patched to call WhisperForConditionalGeneration.generate()
        unwrapped = self.accelerator.unwrap_model(model)

        # FIX-9: bf16=True only activates autocast during training.
        # During eval, compute_loss_context_manager() returns nullcontext(), so
        # float32 input_features collide with bfloat16 model weights in conv1d:
        #   RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
        # Fix: cast explicitly to the model's dtype — no reliance on autocast at all.
        model_dtype = next(unwrapped.parameters()).dtype   # bfloat16
        input_features = inputs["input_features"].to(model_dtype)

        with torch.no_grad():
            generated_tokens = unwrapped.generate(
                input_features,
                **gen_kwargs,
            )

        if has_labels:
            labels = inputs["labels"]
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )
            with torch.no_grad():
                outputs = model(
                    input_features=input_features,   # already cast above
                    labels=labels,
                )
            loss = outputs.loss.mean().detach()
        else:
            loss   = None
            labels = None

        if labels is not None and labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

    # ------------------------------------------------------------------
    def _pad_tensors_to_max_len(self, tensor: torch.Tensor, max_length: int) -> torch.Tensor:
        # Robust lookup: works whether self.tokenizer / self.processing_class
        # is a WhisperProcessor, WhisperTokenizer, or FeatureExtractor.
        # WhisperProcessor delegates .pad_token_id to its tokenizer via __getattr__.
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        pad_token_id = getattr(tok, "pad_token_id", None) or 0
        padded = torch.full(
            (tensor.shape[0], max_length),
            pad_token_id,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, : tensor.shape[-1]] = tensor
        return padded


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def build_compute_metrics(tokenizer: WhisperTokenizer):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # FIX-3: predictions can be (token_ids, scores) when beams > 1 or
        # return_dict_in_generate=True
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = [p.strip() for p in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
        label_str = [l.strip() for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    return compute_metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    set_seed(SEED)

    global_rank = int(os.environ.get("RANK", 0))
    is_main     = global_rank == 0

    if not is_main:
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"]   = "disabled"

    if is_main:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model":          MODEL_ID,
                "lora_r":         LORA_CONFIG.r,
                "lora_alpha":     LORA_CONFIG.lora_alpha,
                "lora_dropout":   LORA_CONFIG.lora_dropout,
                "target_modules": LORA_CONFIG.target_modules,
                "lora_scope":     "encoder+decoder",
                "language":       LANGUAGE,
                "task":           TASK,
                "train_chunks":   39928,
                "val_chunks":     5110,
                "test_chunks":    5166,
            },
        )
# INH100017020_E
    # ── Processor ──────────────────────────────────────────────────────
    logger.info("Loading processor …")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    tokenizer = processor.tokenizer

    # ── Base model ─────────────────────────────────────────────────────
    logger.info("Loading base model …")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.config.use_cache          = False   # required for gradient checkpointing

    # ── Apply LoRA ──────────────────────────────────────────────────────
    logger.info("Applying LoRA (encoder + decoder) …")
    model = get_peft_model(model, LORA_CONFIG)
    assert_encoder_and_decoder(model)

    # ── FIX-1: patch forward / generate for Whisper + transformers-5.x ─
    patch_peft_whisper_forward(model)

    if is_main:
        log_trainable_parameters(model)
        model.print_trainable_parameters()

    # ── Gradient checkpointing ──────────────────────────────────────────
    # Must come AFTER patch (patch operates on the PEFT wrapper;
    # gradient_checkpointing_enable() modifies the inner encoder + decoder
    # layers — both are orthogonal and order does not matter, but this is
    # cleaner).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Dataset ────────────────────────────────────────────────────────
    logger.info("Loading manifests …")
    raw_datasets = load_manifests()

    logger.info("Preprocessing …")
    processed = preprocess_dataset(raw_datasets, processor)

    train_dataset = processed["train"]
    eval_dataset  = processed["validation"]
    test_dataset  = processed["test"]

    # ── Collator ───────────────────────────────────────────────────────
    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training arguments ─────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(**TRAINING_ARGS)

    # ── Metrics ────────────────────────────────────────────────────────
    compute_metrics = build_compute_metrics(tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = WhisperLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,        # FIX-2: full processor, not just feature_extractor
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [
            os.path.join(OUTPUT_DIR, d)
            for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if ckpts:
            # FIX-5: sort by step number embedded in directory name;
            # ctime is unreliable on NFS / remote filesystems.
            checkpoint = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    # ── Train ──────────────────────────────────────────────────────────
    logger.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if is_main:
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    # ── Test evaluation ────────────────────────────────────────────────
    logger.info("Running test evaluation …")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    if is_main:
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info(f"Test metrics: {test_metrics}")

    if is_main and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()