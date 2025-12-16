# llama.cpp Backend Guide

This guide explains how to use Cordon with the llama.cpp backend for GPU-accelerated log analysis.

## Overview

Cordon supports two embedding backends:

1. **sentence-transformers** (default): Uses PyTorch, best for native installations
2. **llama.cpp**: Uses GGUF models, enables GPU acceleration in containers via Vulkan

## Why llama.cpp?

### Advantages

- **GPU in Containers**: Works with Vulkan device passthrough (MPS doesn't work in containers)
- **Lower Memory**: GGUF quantization reduces model size (~21MB vs ~100MB)
- **Cross-Platform**: Same backend works on Linux, macOS containers, and Windows

### Trade-offs

- **No Batching**: Processes embeddings one at a time
- **Performance**: See benchmarks below - competitive with sentence-transformers on CPU, faster with GPU acceleration

### Recommendation

- **Native installations**: Use sentence-transformers (default) for simplicity
- **Container deployments**: Use llama.cpp - enables GPU acceleration via Vulkan

## Performance Benchmarks

Tested on 2,000-line Apache log file (macOS, Podman machine with libkrun):

| Backend | Mode | Time | Blocks Detected | Notes |
|---------|------|------|-----------------|-------|
| sentence-transformers | CPU | 27.42s | 48 | Baseline (PyTorch) |
| llama.cpp | CPU | 24.04s | 47 | 12% faster than baseline |
| llama.cpp | GPU (Vulkan) | 22.20s | 47 | 19% faster than baseline, 8% faster than llama-cpp CPU |

**Key Findings:**
- llama.cpp CPU is 12% faster than sentence-transformers despite no batching
- GPU acceleration provides an additional 8% speedup over CPU
- CPU-only mode is fully functional for environments without GPU

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
cordon --backend llama-cpp /path/to/logfile.log

# GPU-accelerated (offload 10 layers)
cordon --backend llama-cpp \
    --n-gpu-layers 10 \
    /path/to/logfile.log

# All GPU layers (maximum acceleration)
cordon --backend llama-cpp \
    --n-gpu-layers -1 \
    /path/to/logfile.log
```

### Custom Model

```bash
cordon --backend llama-cpp \
    --model-path ~/models/custom.gguf \
    --n-gpu-layers 10 \
    /path/to/logfile.log
```

### Advanced Options

```bash
cordon --backend llama-cpp \
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
# Note: GPU passthrough is enabled by default with libkrun
# The --device /dev/dri flag is optional
podman run -v ./logs:/logs cordon:latest \
    --backend llama-cpp \
    --n-gpu-layers 10 \
    /logs/system.log

# Linux with NVIDIA (CUDA)
podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    -v ./logs:/logs:z cordon:latest \
    --backend llama-cpp \
    --n-gpu-layers 10 \
    /logs/system.log

# Linux with AMD/Intel (Vulkan)
podman run --device /dev/dri -v ./logs:/logs:z cordon:latest \
    --backend llama-cpp \
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
cordon --backend llama-cpp --model-path ./all-MiniLM-L6-v2-Q4_K_M.gguf logs/system.log
```
