# llama.cpp Backend Guide

This guide explains how to use Cordon with the llama.cpp backend for GPU-accelerated log analysis in containerized environments.

## Table of Contents

- [Overview](#overview)
- [Why llama.cpp?](#why-llamacpp)
- [Installation](#installation)
- [Model Preparation](#model-preparation)
- [Usage](#usage)
- [Container Deployment](#container-deployment)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

Cordon supports two embedding backends:

1. **sentence-transformers** (default): PyTorch-based, requires ~2GB RAM, CPU-optimized
2. **llama.cpp**: GGUF-based, supports GPU acceleration via Vulkan, optimized for containers

The llama.cpp backend enables GPU acceleration in Linux containers on macOS via Vulkan passthrough, providing 2-3x speedup over CPU-only processing.

## Why llama.cpp?

### Advantages

- **GPU Acceleration in Containers**: Works with Vulkan device passthrough (`--device /dev/dri`)
- **Lower Memory Footprint**: GGUF quantization reduces model size
- **Better Container Performance**: Vulkan works in Linux containers on macOS (MPS does not)
- **Cross-Platform**: Same backend works on Linux, macOS containers, and Windows

### Trade-offs

- **No Batching**: Processes embeddings one at a time (vs. sentence-transformers batching)
- **Slightly Lower Throughput**: ~10-15% slower than sentence-transformers with batching on CPU
- **Model Compatibility**: Works with GGUF format models (auto-downloads default, or convert custom models)

### Detailed Comparison

| Feature | sentence-transformers (Default) | llama.cpp Backend |
|---------|---------------------------------|-------------------|
| **Architecture** | PyTorch + HuggingFace | llama.cpp + GGUF |
| **Precision** | FP32 (Full Precision) | Quantized (Q4_K_M, 4-bit) |
| **Throughput** | High (Batched processing, batch_size=32) | Low (Sequential, batch_size=1 only) |
| **Context Window** | Model dependent (512 tokens for all-MiniLM-L6-v2) | 512 tokens (for default model) |
| **Truncation** | Automatic (via Tokenizer) | Automatic (via Model Context) |
| **Normalization** | L2 Normalized | L2 Normalized |
| **Memory Usage** | High (~2GB for small model) | Low (~25MB for Q4_K_M) |

**Critical Limitation - No Batching Support**:

The llama.cpp backend with BERT-based embedding models (like all-MiniLM-L6-v2) **does not support batching** in the current llama-cpp-python implementation. When you pass multiple texts to `create_embedding()`, it fails with `llama_decode returned -1`. This is a known limitation of BERT architecture support in llama.cpp.

**Impact**:
- Must use `batch_size=1` (sequential processing)
- Significantly slower than sentence-transformers (~2.5x slower on CPU, ~12x slower vs GPU batching)
- The `--batch-size` parameter is effectively ignored for llama.cpp backend

**Alternative Models Tested**:
- **nomic-embed-text-v1.5** (8192 context): Also fails with batching, 12x slower than sentence-transformers
- **F32 variants**: Same batching limitation, larger file size with minimal accuracy gain

**Recommendation**: Use sentence-transformers for native installations. Use llama.cpp only for containerized deployments where PyTorch MPS is unavailable.

## Installation

### Local Installation

Install llama.cpp support as an optional dependency:

```bash
# With uv (recommended)
uv pip install 'cordon[llama-cpp]'

# Or with pip
pip install 'cordon[llama-cpp]'
```

### Container Installation

The provided Containerfile includes llama.cpp support with Vulkan acceleration:

```bash
# Build container with llama.cpp support
podman build -t cordon:latest -f Containerfile .

# The container includes:
# - llama-cpp-python compiled with Vulkan support
# - Vulkan drivers and loaders
# - Both backends (sentence-transformers and llama.cpp)
```

## Model Preparation

### Option 1: Auto-Download (Recommended)

**Cordon automatically downloads the default GGUF model on first use**, matching the sentence-transformers backend behavior. No manual preparation needed!

```bash
# First run will automatically download the model (~21MB)
cordon --backend llama-cpp --use-faiss /path/to/logfile.log

# Output:
# Downloading default GGUF model: all-MiniLM-L6-v2-Q4_K_M.gguf
# Using HuggingFace Hub (same as sentence-transformers)
# This is a one-time download (~21MB)...
# ✓ Model downloaded successfully
#   Cache: ~/.cache/huggingface/hub/...
```

The model is cached in `~/.cache/huggingface/` (shared with sentence-transformers) and reused for all future runs.

### Option 2: Manual Download (Optional)

For offline/airgapped environments or custom models, download manually:

```bash
# Download from HuggingFace
cd ~/.cache/huggingface/hub
wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf

# Or specify custom model path
cordon --backend llama-cpp --use-faiss --model-path /path/to/model.gguf /path/to/logfile.log
```

See [HuggingFace model repository](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF) for all available quantizations.

### Option 3: Convert Your Own Model

Convert sentence-transformers models to GGUF format:

```bash
# Install conversion tools
pip install transformers torch

# Convert model (requires llama.cpp repository)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Run conversion script
python convert-hf-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --outfile ~/custom-models/all-MiniLM-L6-v2.gguf \
    --outtype f16
```

**Note**: Not all sentence-transformers models are compatible with llama.cpp. Models based on BERT, RoBERTa, or similar architectures work best.

### Quantization (Optional)

Reduce model size with quantization:

```bash
# Quantize to Q4_K_M (recommended balance of size/quality)
./quantize ~/custom-models/all-MiniLM-L6-v2.gguf \
    ~/custom-models/all-MiniLM-L6-v2-Q4_K_M.gguf \
    Q4_K_M
```

Common quantization levels:
- `f16`: Full 16-bit precision (largest, highest quality)
- `Q8_0`: 8-bit quantization (good balance)
- `Q4_K_M`: 4-bit quantization (smallest, still good quality)
- `Q2_K`: 2-bit quantization (very small, noticeable quality loss)

## Usage

### Basic Usage

```bash
# CPU-only mode with auto-download (default, recommended)
cordon --backend llama-cpp --use-faiss /path/to/logfile.log

# GPU-accelerated (offload 10 layers to GPU)
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    /path/to/logfile.log

# All GPU layers (maximum GPU acceleration)
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers -1 \
    /path/to/logfile.log

# Custom model path (for offline/custom models)
cordon --backend llama-cpp \
    --use-faiss \
    --model-path ~/custom-models/my-model.gguf \
    /path/to/logfile.log
```

### Advanced Options

```bash
# With custom model and advanced parameters
cordon --backend llama-cpp \
    --use-faiss \
    --model-path ~/custom-models/my-model.gguf \
    --n-gpu-layers 10 \           # GPU layer offloading
    --n-ctx 2048 \                # Context size
    --n-threads 8 \               # CPU threads
    --window-size 20 \            # Analysis window size
    --anomaly-percentile 0.05 \   # Top 5% anomalies
    --detailed \                  # Show statistics
    /path/to/logfile.log

# With auto-download and advanced parameters (no model path needed)
cordon --backend llama-cpp \
    --use-faiss \
    --n-gpu-layers 10 \
    --window-size 20 \
    --anomaly-percentile 0.05 \
    --detailed \
    /path/to/logfile.log
```

### Backend Comparison

Run the same analysis with both backends to compare:

```bash
# sentence-transformers (default)
cordon --detailed /path/to/logfile.log

# llama.cpp (auto-download)
cordon --backend llama-cpp \
    --use-faiss \
    --detailed \
    /path/to/logfile.log
```

## Container Deployment

### Building the Container

```bash
# Build with llama.cpp support
podman build -t cordon:latest -f Containerfile .

# Verify build includes llama.cpp
podman run cordon:latest python3.13 -c "import llama_cpp; print('llama.cpp OK')"
```

### Running with GPU Acceleration

```bash
# GPU-accelerated analysis (Vulkan via libkrun)
podman run --device /dev/dri \
    -v ./logs:/logs:ro \
    -v ./models:/models:ro \
    cordon:latest \
    --backend llama-cpp \
    --use-faiss \
    --model-path /models/model.gguf \
    --n-gpu-layers 10 \
    /logs/system.log

# CPU-only analysis
podman run \
    -v ./logs:/logs:ro \
    -v ./models:/models:ro \
    cordon:latest \
    --backend llama-cpp \
    --use-faiss \
    --model-path /models/model.gguf \
    /logs/system.log
```

### Volume Mounts

- `/logs`: Log files to analyze (read-only recommended)
- `/models`: GGUF model files (read-only recommended)

### GPU Access Requirements

For GPU acceleration in containers:

1. **Podman with libkrun**: Provides `/dev/dri` device for Vulkan
2. **Vulkan drivers**: Included in container
3. **Device passthrough**: `--device /dev/dri` flag

Check libkrun status:

```bash
podman machine list
# Look for "VM TYPE libkrun"
```

## Performance

### Benchmarks

Real-world performance on Apple M2 Pro (10-core CPU) with **Apache 2k.log** (1999 lines, 167KB, 400 windows):

#### Native Performance (macOS)

| Backend | Configuration | Processing Time | Relative Speed | Notes |
|---------|---------------|-----------------|----------------|-------|
| **sentence-transformers** | MPS GPU (default) | **3.56s** | **1.0x** ✅ | **Best for native** |
| **llama.cpp** | GPU (10 layers) | 7.65s | 0.47x | Slower (no batching) |

**Key Findings:**
- **sentence-transformers wins on native**: Batching advantage outweighs GPU benefits for medium/large logs
- **llama.cpp is slower natively**: Sequential processing cannot compete with batched transformers
- **Recommendation**: Use **sentence-transformers (default)** for native macOS/Linux workloads

#### Container Performance (Podman + libkrun on macOS)

| Backend | Configuration | Processing Time | Relative Speed | Notes |
|---------|---------------|-----------------|----------------|-------|
| **llama.cpp** | GPU (10 layers, Vulkan) | **17.56s** | **1.0x** ✅ | **Best for containers** |
| **llama.cpp** | CPU-only | 19.15s | 0.92x | Minimal GPU benefit |

**Container Findings:**
- **Vulkan provides ~9% speedup**: Container GPU overhead limits benefits
- **sentence-transformers unavailable**: MPS cannot work in Linux containers
- **Recommendation**: Use **llama.cpp with GPU** for containers, but native is 5x faster

*Results vary based on model size, quantization, hardware, and log file size.*

### Optimization Tips

1. **Backend Selection (Most Critical!)**:
   - **Native workloads**: Use **sentence-transformers (default)** - 2x faster than llama.cpp
   - **Containers**: Use **llama.cpp with GPU** - only option since MPS doesn't work in containers
   - **Production**: Prefer native installation over containers (5x faster)

2. **GPU Layers** (for llama.cpp only):
   - In containers: Always use `--n-gpu-layers 10` or higher (~9% speedup over CPU)
   - Native: GPU helps but batching in sentence-transformers is faster overall
   - Avoid CPU-only mode (0 layers) - minimal benefit over GPU

3. **Quantization**: Use Q4_K_M for best size/quality trade-off
4. **Batch Processing**: Process multiple logs sequentially to amortize model load time
5. **Context Size**: Reduce `--n-ctx` if you don't need large context windows

**Performance Summary**:
- **Best overall**: sentence-transformers on native macOS/Linux (3.56s)
- **Best for containers**: llama.cpp with GPU (17.56s)
- **Avoid**: llama.cpp on native (7.65s) - use sentence-transformers instead

### GPU Layer Selection

The optimal number of GPU layers depends on your hardware:

```bash
# Test different layer counts (uses auto-downloaded model)
for layers in 0 5 10 20 -1; do
    echo "Testing $layers GPU layers..."
    time cordon --backend llama-cpp \
        --use-faiss \
        --n-gpu-layers $layers \
        /path/to/logfile.log
done
```

## Troubleshooting

### Import Error

**Error**: `ImportError: llama-cpp-python is required`

**Solution**:
```bash
uv pip install 'cordon[llama-cpp]'
# or
pip install llama-cpp-python
```

### Model Not Found

**Error**: `ValueError: GGUF model file not found`

**Solution**:
- Verify file path with `ls -l /path/to/model.gguf`
- Use absolute paths or paths relative to current directory
- Check file permissions (must be readable)

### Invalid GGUF File

**Error**: `RuntimeError: Failed to load GGUF model`

**Solution**:
- Verify file is valid GGUF format: `file model.gguf`
- Re-download or re-convert the model
- Check model was created with embedding support

### GPU Not Detected

**Symptom**: Performance similar to CPU-only mode

**Debugging**:
```bash
# Check Vulkan device availability
podman run --device /dev/dri cordon:latest vulkaninfo

# Verify libkrun is enabled
podman machine list

# Check llama.cpp can see GPU
podman run --device /dev/dri cordon:latest \
    python3.13 -c "from llama_cpp import Llama; print('GPU support OK')"
```

**Solutions**:
- Ensure `--device /dev/dri` flag is used
- Verify libkrun machine is running
- Check Vulkan drivers are installed in container

### Out of Memory

**Error**: `RuntimeError: Failed to allocate memory`

**Solutions**:
1. Reduce GPU layers: `--n-gpu-layers 5`
2. Use smaller quantization: Q4_K_M or Q2_K
3. Reduce context size: `--n-ctx 1024`
4. Increase container memory limit

### Slow Performance

**Symptoms**: Much slower than expected

**Checklist**:
- [ ] Using GPU acceleration (`--n-gpu-layers > 0`)
- [ ] GPU device is passed through (`--device /dev/dri`)
- [ ] Model is quantized (Q4_K_M recommended)
- [ ] Not running other GPU-intensive processes
- [ ] Container has sufficient CPU/memory resources

## Best Practices

1. **Start Simple**: Begin with CPU-only mode to verify model works
2. **Incremental GPU**: Add GPU layers gradually to find optimal setting
3. **Model Selection**: Use Q4_K_M quantization for production
4. **Container Optimization**: Pre-load models in container image for faster startup
5. **Logging**: Use `--detailed` flag to monitor performance metrics

## See Also

- [Container Usage Guide](CONTAINER.md) - Container deployment details
- [README](../README.md) - General Cordon documentation
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp) - Upstream project

## Support

For issues specific to llama.cpp backend:

1. Check this troubleshooting guide
2. Verify model compatibility
3. Test with CPU-only mode first
4. Report issues with full error messages and system info
