# Containerfile for Cordon CLI
# Semantic anomaly detection for system logs
# Base: CentOS Stream 10 with Python 3.13

# Use CentOS Stream 10 as base image
FROM quay.io/centos/centos:stream10-development

# Metadata
LABEL maintainer="Cordon Project"
LABEL description="Cordon - Semantic anomaly detection for system logs"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/root/.cache/torch \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# Install EPEL repository for Python 3.13 and system dependencies
RUN dnf install -y \
    epel-release \
    && dnf install -y \
    python3.13 \
    python3.13-devel \
    python3.13-pip \
    gcc \
    gcc-c++ \
    make \
    cmake \
    git \
    flexiblas-devel \
    openblas \
    openblas-openmp \
    ca-certificates \
    mesa-libGL \
    mesa-vulkan-drivers \
    vulkan-loader \
    vulkan-headers \
    vulkan-devel \
    glslc \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Create application directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Upgrade pip and install build tools
RUN python3.13 -m pip install --upgrade pip setuptools wheel

# Install the application in editable mode with all optional dependencies
# This installs all dependencies from pyproject.toml including llama-cpp
RUN python3.13 -m pip install -e .

# Install llama-cpp-python with architecture-appropriate GPU support
# Automatically detects architecture and builds appropriate backends:
# - x86_64: Vulkan + CUDA (NVIDIA, AMD, Intel GPU support)
# - aarch64: Vulkan only (Apple Silicon, ARM GPUs)
# - CPU fallback: Works on all platforms without GPU
# Build with reduced optimization (-O1) to lower memory usage during compilation
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Building for x86_64: Vulkan + CUDA support" && \
        CMAKE_ARGS="-DGGML_VULKAN=on -DGGML_CUDA=on -DCMAKE_C_FLAGS=-O1 -DCMAKE_CXX_FLAGS=-O1" \
        CMAKE_BUILD_PARALLEL_LEVEL=2 \
        python3.13 -m pip install 'llama-cpp-python>=0.3.0' --no-cache-dir; \
    else \
        echo "Building for $ARCH: Vulkan only" && \
        CMAKE_ARGS="-DGGML_VULKAN=on -DCMAKE_C_FLAGS=-O1 -DCMAKE_CXX_FLAGS=-O1" \
        CMAKE_BUILD_PARALLEL_LEVEL=2 \
        python3.13 -m pip install 'llama-cpp-python>=0.3.0' --no-cache-dir; \
    fi

# Pre-download both default models for offline usage and faster first-run
# Both backends use huggingface_hub for auto-download with shared caching
RUN python3.13 -c "from sentence_transformers import SentenceTransformer; \
from huggingface_hub import hf_hub_download; \
print('Downloading sentence-transformers model...'); \
model = SentenceTransformer('all-MiniLM-L6-v2'); \
print('Downloading GGUF model for llama.cpp backend...'); \
gguf_path = hf_hub_download(repo_id='second-state/All-MiniLM-L6-v2-Embedding-GGUF', filename='all-MiniLM-L6-v2-Q4_K_M.gguf'); \
print(f'Models cached successfully in HuggingFace cache: {gguf_path}')"

# Create a non-root user for running the application (optional, but recommended)
RUN useradd -m -u 1000 cordon && \
    chown -R cordon:cordon /app /root/.cache

# Switch to non-root user
USER cordon

# Set working directory for log file processing
WORKDIR /logs

# Expose the CLI via entrypoint
# Users must provide log file as argument
ENTRYPOINT ["cordon"]

# Default command shows help if no arguments provided
CMD ["--help"]

# Usage examples:
# Build: podman build -t cordon:latest -f Containerfile .
#
# CPU-only sentence-transformers (default):
#   podman run -v ./logs:/logs cordon:latest /logs/system.log
#
# CPU-only llama.cpp (auto-downloads model from HuggingFace cache):
#   podman run -v ./logs:/logs cordon:latest --backend llama-cpp --use-faiss /logs/system.log
#
# GPU on macOS (Vulkan via libkrun):
#   podman run --device /dev/dri -v ./logs:/logs cordon:latest --backend llama-cpp --use-faiss --n-gpu-layers 10 /logs/system.log
#
# GPU on Linux with NVIDIA (CUDA):
#   podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ -v ./logs:/logs:z cordon:latest --backend llama-cpp --use-faiss --n-gpu-layers 10 /logs/system.log
#
# GPU on Linux with AMD/Intel (Vulkan):
#   podman run --device /dev/dri -v ./logs:/logs:z cordon:latest --backend llama-cpp --use-faiss --n-gpu-layers 10 /logs/system.log
#
# Custom model: podman run -v ./logs:/logs -v ./models:/models:ro cordon:latest --backend llama-cpp --use-faiss --model-path /models/custom.gguf /logs/system.log
# Interactive: podman run -it --entrypoint /bin/bash cordon:latest
