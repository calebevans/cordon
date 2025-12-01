FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake git \
    libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN uv pip install --system -e ".[llama-cpp]"

RUN CMAKE_ARGS="-DGGML_VULKAN=on" \
    uv pip install --system --no-cache-dir llama-cpp-python

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q4_K_M.gguf')"

WORKDIR /logs

ENTRYPOINT ["cordon"]
CMD ["--help"]
