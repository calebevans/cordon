FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake git \
    libvulkan1 libvulkan-dev vulkan-tools glslang-tools glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/cordon && touch src/cordon/__init__.py

RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

RUN CMAKE_ARGS="-DGGML_VULKAN=on" CMAKE_BUILD_PARALLEL_LEVEL=2 \
    uv pip install --system ".[llama-cpp]"

COPY src/ ./src/
RUN uv pip install --system --no-deps .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q4_K_M.gguf')"

WORKDIR /logs

ENTRYPOINT ["cordon"]
CMD ["--help"]
