# llama.cpp Backend Guide

This guide explains how to use Cordon with the llama.cpp backend for GPU-accelerated log analysis.

## Overview

Cordon supports two embedding backends:

1. **sentence-transformers** (default): PyTorch-based, best for native installations
2. **llama.cpp**: GGUF-based, enables GPU acceleration in containers via Vulkan

## Why llama.cpp?

### Advantages

- **GPU in Containers**: Works with Vulkan device passthrough (MPS doesn't work in containers)
- **Lower Memory**: GGUF quantization reduces model size (~21MB vs ~100MB)
- **Cross-Platform**: Same backend works on Linux, macOS containers, and Windows

### Trade-offs

- **No Batching**: Processes embeddings one at a time (slower than sentence-transformers on CPU)
- **Performance**: ~10-15% slower than sentence-transformers with batching on CPU

### Recommendation

- **Native installations**: Use sentence-transformers (default) - 2x faster
- **Container deployments**: Use llama.cpp - only option for GPU acceleration

## Installation

### Local

```bash
# With uv
uv pip install 'cordon[llama-cpp]'

# Or with pip
pip install 'cordon[llama-cpp]'
```

### Container

The Containerfile includes llama.cpp with Vulkan support:

```bash
make container-build
```

## Usage

### Auto-Download (Recommended)

The default GGUF model is automatically downloaded on first use:

```bash
# CPU-only
cordon --backend llama-cpp --use-faiss /path/to/logfile.log

# GPU-accelerated (offload 10 layers)
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    /path/to/logfile.log

# All GPU layers (maximum acceleration)
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers -1 \
    /path/to/logfile.log
```

### Custom Model

```bash
cordon --backend llama-cpp \
    --use-faiss \
    --model-path ~/models/custom.gguf \
    --n-gpu-layers 10 \
    /path/to/logfile.log
```

### Advanced Options

```bash
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \           # GPU layer offloading
    --n-ctx 2048 \                # Context size
    --n-threads 8 \               # CPU threads
    --window-size 20 \
    --detailed \
    /path/to/logfile.log
```

## Container Deployment

### Build

```bash
make container-build
```

### Run with GPU

```bash
# macOS with libkrun (Vulkan)
podman run --device /dev/dri -v ./logs:/logs cordon:latest \
    --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    /logs/system.log

# Linux with NVIDIA (CUDA)
podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    -v ./logs:/logs:z cordon:latest \
    --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    /logs/system.log

# Linux with AMD/Intel (Vulkan)
podman run --device /dev/dri -v ./logs:/logs:z cordon:latest \
    --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    /logs/system.log
```

## Model Information

### Default Model

- **Name**: all-MiniLM-L6-v2-Q4_K_M.gguf
- **Size**: ~21MB
- **Context**: 512 tokens
- **Download**: Automatic via HuggingFace Hub
- **Cache**: `~/.cache/huggingface/`

### Manual Download

```bash
# From HuggingFace
wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf

# Use with Cordon
cordon --backend llama-cpp --use-faiss --model-path ./all-MiniLM-L6-v2-Q4_K_M.gguf logs/system.log
```
