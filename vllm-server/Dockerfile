# Simple Jev classifier API on vLLM, as a custom container for Hugging Face
# Inference Endpoints (weights mounted at /repository) or any CUDA host.
#
#   docker build --platform linux/amd64 -f vllm-server/Dockerfile -t ghcr.io/<you>/simple-jev:vllm-<tag> .
#
# Environment: see vllm-server/jev_vllm.py (MODEL, MODEL_ALIAS, TEMPLATE, PORT,
# MAX_MODEL_LEN, MAX_IMAGES, GPU_MEMORY_UTILIZATION, TOP_LOGPROBS, VLLM_EXTRA_ARGS).
FROM vllm/vllm-openai:v0.29.0

WORKDIR /app
COPY common ./common
COPY vllm-server ./vllm-server
RUN pip install --no-cache-dir "pydantic>=2" fastapi uvicorn httpx

ENV MODEL=/repository TEMPLATE=v2c PORT=8000 VLLM_PORT=8001 MAX_MODEL_LEN=8192 \
    MAX_IMAGES=4 TOP_LOGPROBS=64 ENABLE_OPEN_JEV_ADVANCED_METRICS=1 PYTHONUNBUFFERED=1
EXPOSE 8000

# The adapter starts vLLM itself; /health is 503 until vLLM serves.
ENTRYPOINT []
CMD ["python3", "/app/vllm-server/jev_vllm.py"]
