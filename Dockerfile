# Simple Jev HF server as a custom container for Hugging Face Inference
# Endpoints (or any CUDA host). Weights are NOT baked in: Inference Endpoints
# mounts the selected model repository at /repository; elsewhere set MODEL to
# a Hub id or a mounted directory.
#
#   docker build --platform linux/amd64 -t ghcr.io/<you>/simple-jev:<tag> .
#
# Environment (all optional):
#   MODEL          model path or Hub id (default /repository)
#   MODEL_ALIAS    extra name accepted in the request "model" field, e.g. the
#                  Hub id, so clients do not have to send "/repository"
#   DTYPE          bfloat16 (default) | float16 | float32
#   MAX_MODEL_LEN  per-branch token limit (default 8192)
#   MAX_IMAGES     images per request (default 4)
#   PORT           listen port (default 8000)
#
# The server needs Python >= 3.12, which the pytorch/pytorch images do not
# ship, so this starts from python:3.12 and installs the CUDA 12.8 torch
# wheels; they bundle the CUDA runtime libraries, so no CUDA base is needed.
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
WORKDIR /app

RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

COPY common ./common
COPY hf-server ./hf-server
RUN pip install ./hf-server pillow

ENV MODEL=/repository DTYPE=bfloat16 MAX_MODEL_LEN=8192 MAX_IMAGES=4 PORT=8000 \
    ENABLE_OPEN_JEV_ADVANCED_METRICS=1
EXPOSE 8000

# The server binds its port only after the weights are loaded, so /health
# answering 200 means ready. --device cuda:0 keeps accelerate from sharding.
CMD ["sh", "-c", "exec simple-jev --model \"$MODEL\" ${MODEL_ALIAS:+--model-alias \"$MODEL_ALIAS\"} --device cuda:0 --dtype \"$DTYPE\" --max-model-len \"$MAX_MODEL_LEN\" --max-images \"$MAX_IMAGES\" --host 0.0.0.0 --port \"$PORT\""]
