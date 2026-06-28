# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Hindi Singing ASR — GPU inference image
#
# Uses an NVIDIA CUDA runtime base so torch can see the H100. The model weights
# are NOT baked in; mount them at runtime (see README) so the image stays slim
# and the same image works for any checkpoint.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

# System deps: python + ffmpeg/libsndfile for audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg libsndfile1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch (CUDA 12.4 wheels) first, then the rest of the requirements.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt

# Application code.
COPY app.py transliterate.py ./
COPY static ./static

# Defaults; override at `docker run` time as needed.
ENV MODEL_PATH=/models/whisper-large-v3-turbo-merged \
    IDLE_TIMEOUT_SECONDS=300 \
    HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
