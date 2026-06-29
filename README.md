# Hindi Singing ASR.

Fine-tuning Whisper models for **Hindi song lyrics transcription**.

**Project:** Saregama × IIT Bombay

---

## Overview

This repository contains the complete training and inference pipeline for Automatic Speech Recognition (ASR) on Hindi singing voice.

The project includes:

* Vocal extraction from songs using Demucs
* LoRA fine-tuning of Whisper models
* Multi-GPU distributed training
* Greedy and Beam Search inference
* N-best hypothesis generation for rescoring experiments

---
<h2>Tech Stack</h2>

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/TorchAudio-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PEFT-LoRA-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accelerate-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Datasets-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Demucs-Vocal_Separation-2E86DE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/NVIDIA-H100-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/W%26B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black"/>
</p>

# Installation

Clone the repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

Run all commands from the repository root:

```text
hindi-song-asr/
```

---

# Repository Structure

```text
hindi-song-asr/
│
├── data/
│   ├── downloads/
│   │   ├── audio/
│   │   └── vocals/
│   │
│   └── training_data_final/
│       ├── manifests/
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   └── test.jsonl
│       │
│       ├── chunks/
│       │   ├── INH100017020_E/
│       │   │   ├── 00001.wav
│       │   │   ├── 00002.wav
│       │   │   └── ...
│       │
│       ├── clean_dataset.json
│       └── split.json
│
├── data_prep/
├── training_scripts/
├── inference/
├── outputs (to be added by user, Model goes here)/
└── predictions (to be added by user, predictions come here)/
```

---

# Data Format

## Manifest Files

Training and inference use JSONL manifest files located in:

```text
data/training_data_final/manifests/
```

Each line contains a single sample:

```json
{
  "audio": "data/training_data_final/chunks/SONG_ID/00001.wav",
  "text": "transcript"
}
```

The audio path must point to an existing WAV chunk.

---

## Other Dataset Files

### `clean_dataset.json`

Contains the original lyrics along with timestamps.

### `split.json`

Defines the train, validation, and test song splits.

---

# Scripts

## 1. Vocal Extraction

### Script

```text
data_prep/dataset/vocals.py
```

### Purpose

Extracts isolated vocal tracks from MP3 songs using **Demucs**.

### Input

```text
data/downloads/audio/
```

### Output

```text
data/downloads/vocals/
```

### Requirements

* MP3 files
* `demucs` installed

### Run

```bash
python data_prep/dataset/vocals.py
```

---

# 2. Training

### Script

```text
training_scripts/train_whisper_medium_encdec_lora.py
```

### Purpose

Performs LoRA fine-tuning of **Whisper Large-v3 Turbo** using Distributed Data Parallel (DDP).

Both encoder and decoder attention layers are trained.

### Run

```bash
torchrun --nproc_per_node=2 \
training_scripts/train_whisper_medium_encdec_lora.py
```

### Requirements

Manifest files:

```text
data/training_data_final/manifests/

train.jsonl
val.jsonl
test.jsonl
```

Each manifest entry must contain

```json
{
  "audio": "...",
  "text": "..."
}
```

Audio files must exist under

```text
data/training_data_final/chunks/
```

Training logs are tracked using **Weights & Biases (WandB)**.

---

## Training Configuration

| Parameter           | Value                           |
| ------------------- | ------------------------------- |
| Model               | `openai/whisper-large-v3-turbo` |
| Language            | `hi`                            |
| LoRA Rank           | 32                              |
| LoRA Alpha          | 64                              |
| Dropout             | 0.05                            |
| Epochs              | 20                              |
| Batch Size          | 32                              |
| Learning Rate       | 1e-4                            |
| Evaluation Interval | Every 500 steps                 |
| Checkpointing       | Best 3 checkpoints by CER       |

### Output

```text
outputs/
└── whisper-large-v3-turbo-lora/
```

Training metrics are logged to WandB.

---

# 3. Inference (Greedy / Beam Search)

### Script

```text
inference/infer_generic.py
```

### Purpose

Runs inference using a trained LoRA adapter.

Supported decoding strategies:

* Greedy
* Beam Search
* Beam Search + Repetition Penalty

---

## Run

### Greedy

```bash
python inference/infer_generic.py
```

### Beam Search

```bash
python inference/infer_generic.py \
--decode_strategy beam
```

### Custom Model

```bash
python inference/infer_generic.py \
--model_id openai/whisper-large-v3-turbo \
--adapter outputs/whisper-large-v3-turbo-lora \
--language hi
```

---

## Requirements

* `test.jsonl`
* Audio chunks referenced in the manifest
* Trained LoRA adapter

Default adapter:

```text
outputs/whisper-medium-enc-lora
```

---

## Output

```text
predictions/
└── <model>_<strategy>/
    ├── test_predictions_<timestamp>.jsonl
    ├── test_predictions_<timestamp>.txt
    └── test_summary_<timestamp>.json
```

Outputs include:

* Per-sample CER
* Per-sample WER
* Corpus CER/WER
* Inference timing

---

## Command Line Arguments

| Argument            | Default                                         | Description                        |
| ------------------- | ----------------------------------------------- | ---------------------------------- |
| `--model_id`        | `openai/whisper-medium`                         | Base Whisper model                 |
| `--adapter`         | `outputs/whisper-medium-enc-lora`               | LoRA adapter                       |
| `--language`        | `hi`                                            | Language code                      |
| `--manifest`        | `data/training_data_final/manifests/test.jsonl` | Test manifest                      |
| `--decode_strategy` | `greedy`                                        | `greedy`, `beam`, or `beam_reppen` |
| `--batch_size`      | `16`                                            | Batch size                         |
| `--limit`           | `0`                                             | Maximum samples (0 = all)          |

---

# 4. N-best Beam Search Inference

### Script

```text
inference/infer_beam.py
```

### Purpose

Runs N-best beam search decoding and returns multiple hypotheses with their log-probabilities for downstream rescoring.

---

## Run

### Default

```bash
python inference/infer_beam.py
```

### Custom Configuration

```bash
python inference/infer_beam.py \
--model_id openai/whisper-large-v3-turbo \
--adapter outputs/whisper-large-v3-turbo-lora \
--num_beams 10 \
--num_return_sequences 5
```

---

## Requirements

* Test manifest
* Audio chunks
* Trained LoRA adapter

Default adapter:

```text
outputs/whisper-large-v3-turbo-lora
```

---

## Output

```text
predictions/
└── <model>_nbest_b<beams>/
    ├── nbest_<timestamp>.jsonl
    └── nbest_summary_<timestamp>.json
```

The JSONL output contains:

* Multiple hypotheses
* Log probabilities
* Top-1 prediction
* Beam scores

The summary file reports overall CER and WER.

---

## Command Line Arguments

| Argument                 | Default                                         | Description                             |
| ------------------------ | ----------------------------------------------- | --------------------------------------- |
| `--model_id`             | `openai/whisper-large-v3-turbo`                 | Base Whisper model                      |
| `--adapter`              | `outputs/whisper-large-v3-turbo-lora`           | LoRA adapter                            |
| `--language`             | `hi`                                            | Language code                           |
| `--manifest`             | `data/training_data_final/manifests/test.jsonl` | Test manifest                           |
| `--num_beams`            | `5`                                             | Beam width                              |
| `--num_return_sequences` | `5`                                             | Number of returned hypotheses (≤ beams) |
| `--batch_size`           | `32`                                            | Batch size                              |
| `--limit`                | `0`                                             | Maximum samples (0 = all)               |
| `--save_token_ids`       | `False`                                         | Save token IDs in output                |

---

# Outputs

## Training

```text
outputs/
└── whisper-large-v3-turbo-lora/
```

Contains LoRA adapter checkpoints.

---

## Inference

```text
predictions/
```

Contains:

* JSONL predictions
* Human-readable transcripts
* Evaluation metrics
* Corpus-level CER/WER summaries

---

# Notes

* All training and inference use the JSONL manifest files as the primary dataset interface.
* Audio files referenced in the manifests must exist on disk.
* Multi-GPU training is implemented using PyTorch Distributed Data Parallel (DDP).
* Training metrics are automatically logged to Weights & Biases.
