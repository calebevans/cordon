# Cordon

**Semantic anomaly detection for system log files**

Cordon is a Python library that uses transformer-based embeddings and density-based scoring to identify **semantically unusual patterns** in large log files. It's designed to reduce massive logs down to the most anomalous sections for analysis by LLMs or human operators.

**Key principle:** Repetitive patterns (even errors) are considered "normal background." Cordon surfaces **unusual, rare, or clustered events** that stand out semantically from the bulk of the logs.

## Features

- **Semantic Analysis**: Uses transformer models to understand log content meaning, not just keyword matching
- **Density-Based Scoring**: Identifies anomalies using k-NN distance in embedding space
- **Noise Reduction**: Filters out repetitive logs, keeping only unusual patterns

## Installation

### Native Installation (Recommended for macOS GPU Support)

Using `uv` (recommended):

```bash
uv pip install -e .
```

Using pip:

```bash
pip install -e .
```

For development:

```bash
uv pip install -e ".[dev]"
pre-commit install
```

For llama.cpp backend (GPU acceleration in containers):

```bash
# Install with llama.cpp support
uv pip install -e ".[llama-cpp]"
```

For very large logs (optional FAISS support for better performance):

```bash
# For CPU-only systems
uv pip install -e ".[faiss-cpu]"

# For systems with CUDA/GPU
uv pip install -e ".[faiss-gpu]"
```

### Container Installation

For a containerized environment using Podman or Docker:

```bash
# Build container image
./build-container.sh

# Or manually
podman build -t cordon:latest -f Containerfile .
```

**GPU Support in Containers:**
- ✅ **Linux with NVIDIA**: CUDA acceleration via Podman with nvidia-ctk
- ✅ **Linux with AMD/Intel**: Vulkan acceleration via Podman with `--device /dev/dri`
- ✅ **macOS**: Vulkan acceleration via Podman libkrun with `--device /dev/dri`
- ✅ **CPU fallback**: Works on all platforms

See [Container Usage](#container-usage) section below for details.

## Quick Start

### Command Line

```bash
# Basic usage
cordon system.log

# Multiple files
cordon app.log error.log

# With options
cordon --window-size 10 --k-neighbors 10 --anomaly-percentile 0.05 application.log

# Detailed statistics
cordon --detailed --device cuda production.log

# Use FAISS for large logs
cordon --use-faiss --detailed large.log
```

### Python Library

```python
from pathlib import Path
from cordon import SemanticLogAnalyzer, AnalysisConfig

# Basic usage with defaults
analyzer = SemanticLogAnalyzer()
output = analyzer.analyze_file(Path("system.log"))
print(output)

# Advanced configuration
config = AnalysisConfig(
    window_size=10,          # Lines per window
    stride=5,                # Step size for sliding
    k_neighbors=10,          # k-NN parameter
    anomaly_percentile=0.05, # Top 5% most anomalous
    device="cuda"            # Force GPU usage
)

analyzer = SemanticLogAnalyzer(config)
result = analyzer.analyze_file_detailed(Path("application.log"))

print(f"Found {result.merged_blocks} significant blocks")
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Top anomaly score: {result.score_distribution['max']:.4f}")
```

## Backend Options

Cordon supports two embedding backends for flexibility across different deployment environments:

### sentence-transformers (Default)

The default backend using HuggingFace transformers with PyTorch:

```bash
# Automatic GPU detection (MPS on macOS, CUDA on Linux)
cordon system.log

# Force specific device
cordon --device cuda system.log
cordon --device mps system.log
cordon --device cpu system.log
```

**Best for:**
- Native macOS/Linux installations with GPU access
- Development and testing
- Maximum throughput with batching

### llama.cpp Backend

Alternative backend using GGUF models with Vulkan GPU support:

```bash
# CPU-only
cordon --backend llama-cpp \
    --use-faiss \
    --model-path ./models/model.gguf \
    system.log

# GPU-accelerated (offload 10 layers)
cordon --backend llama-cpp \
    --use-faiss \
    --model-path ./models/model.gguf \
    --n-gpu-layers 10 \
    system.log
```

**Best for:**
- **Container deployments** with GPU acceleration via Vulkan (only option for containers)
- Lower memory footprint (GGUF quantization)
- Cross-platform GPU support

**Performance (Apache 2k.log - 1999 lines, 167KB):**
- **Native**: Slower than sentence-transformers (7.65s vs 3.56s) - not recommended
- **Container with GPU**: 17.56s (best option for containers, but native is 5x faster)
- **Container CPU-only**: 19.15s (minimal GPU benefit in containers)

**Critical**: For native macOS/Linux, use sentence-transformers (default) - it's 2x faster due to batching

**Why llama.cpp for containers?**

PyTorch's MPS backend cannot work in Linux containers on macOS. The llama.cpp backend with Vulkan support provides GPU acceleration in containers via device passthrough (`--device /dev/dri`). While GPU provides only ~9% speedup in containers (17.56s vs 19.15s), it's the only option for GPU-accelerated analysis in containerized environments.

See the **[llama.cpp Backend Guide](./docs/llama-cpp-guide.md)** for:
- Model conversion and preparation
- Performance tuning and benchmarks
- Container deployment with GPU
- Troubleshooting and best practices

### Backend Technical Differences

While both backends are unified under the LlamaIndex interface, they have distinct characteristics:

| Feature | sentence-transformers (Default) | llama.cpp Backend |
|---------|---------------------------------|-------------------|
| **Model Precision** | Full precision (FP32) | Quantized (Q4_K_M, 4-bit) |
| **Processing** | Batched (High Throughput) | Sequential (batch_size=1 only) |
| **Token Truncation** | Default tokenizer behavior (usually 512 tokens) | Model context window (512 tokens for default model) |
| **Normalization** | Explicit L2 Normalization | Explicit L2 Normalization |
| **Best For** | Native macOS/Linux, High Performance | Containers, Low Memory, Cross-Platform GPU |

**Note on Batching**: The llama.cpp backend with BERT-based models (like all-MiniLM-L6-v2) **does not support batching** in the current llama-cpp-python implementation. It processes embeddings one at a time (batch_size=1), which significantly impacts performance. This is a known limitation of the BERT architecture support in llama.cpp.

**Note on Truncation**: Both backends now handle token limits automatically via LlamaIndex. Explicit truncation warnings are no longer shown; instead, text exceeding the limit is handled by the underlying model's default behavior (truncation).

## Container Usage

Cordon can be run in a containerized environment using Podman or Docker. This provides a consistent, isolated runtime across platforms.

### Quick Start with Containers

```bash
# Build the container image
./build-container.sh

# Run Cordon on a log file
./run-container.sh /path/to/system.log

# Or use Podman/Docker directly
podman build -t cordon:latest -f Containerfile .
podman run -v $(pwd)/logs:/logs cordon:latest /logs/system.log
```

### GPU Support

The container includes multi-platform GPU support via the llama.cpp backend:

```bash
# macOS with Podman libkrun (Vulkan)
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  --model-path /app/models/all-MiniLM-L6-v2-Q4_K_M.gguf \
  --n-gpu-layers 10 \
  /logs/system.log

# Linux with NVIDIA GPU (CUDA)
podman run --hooks-dir=/usr/share/containers/oci/hooks.d/ \
  -v $(pwd)/logs:/logs:z cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  --model-path /app/models/all-MiniLM-L6-v2-Q4_K_M.gguf \
  --n-gpu-layers 10 \
  /logs/system.log

# Linux with AMD/Intel GPU (Vulkan)
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp \
  --use-faiss \
  --model-path /app/models/all-MiniLM-L6-v2-Q4_K_M.gguf \
  --n-gpu-layers 10 \
  /logs/system.log
```

**Container Performance (Apache 2k.log - 1999 lines):**
- **Vulkan GPU**: 17.56s (~9% faster than CPU)
- **CPU-only**: 19.15s
- **CUDA (NVIDIA)**: Expected 5-10x speedup (hardware dependent, not tested)

**Critical**: Native installation is **5x faster** (3.56s) than containers (17.56s). Use containers only when necessary.

**Note:** sentence-transformers backend with PyTorch MPS cannot work in Linux containers. For best performance on macOS, use native installation with sentence-transformers instead of containers.

### Complete Container Documentation

For detailed container usage, including:
- Podman/Docker installation and setup
- libkrun provider configuration for GPU experimentation
- Build options and customization
- Volume mounting strategies
- Performance comparisons
- Troubleshooting guide

See the **[Container Usage Guide →](./docs/CONTAINER.md)**

## Real-World Examples

Want to see Cordon in action? Check out our **[comprehensive test examples](./docs/test-examples/)** showing results across 9 different log types.

Each example includes:
- Links to original log files from the [LogHub research dataset](https://github.com/logpai/loghub)
- Complete Cordon output with detected anomalies
- Performance metrics and analysis
- Real security incidents and unusual patterns discovered

**Average results**: 2,000-line logs reduced to ~306 lines (84.7% reduction) in ~3 seconds.

**[View Test Examples →](./docs/test-examples/)**

## Primary Use Case: LLM Context Reduction

Cordon is specifically designed to solve the "log file too large for LLM context window" problem:

```
Problem: 50GB production log → Can't fit in any LLM context window
Solution: Cordon → 12 anomalous blocks (few KB) → Send to LLM for analysis
```

The output is intentionally lossy - it **discards repetitive patterns** to focus attention on semantically unusual events. This makes it ideal for LLM pre-processing but unsuitable for comprehensive log analysis.

**Example workflow:**
```python
# Step 1: Extract anomalies
analyzer = SemanticLogAnalyzer()
anomalies = analyzer.analyze_file(Path("production.log"))

# Step 2: Get error counts (traditional tools)
error_count = subprocess.run(["grep", "-c", "ERROR", "production.log"], capture_output=True)

# Step 3: Send curated context to LLM
context = f"""
Log anomalies (top 5% unusual patterns):
{anomalies}

Error summary:
Total ERROR lines: {error_count.stdout.decode()}
(Full error list omitted for brevity)

Question: What caused the outage?
"""
# Now you can fit this in an LLM context window!
```

## How It Works

### Pipeline Architecture

1. **Ingestion**: Read log file line-by-line with UTF-8/latin-1 fallback
2. **Segmentation**: Create overlapping windows of N lines with stride S
3. **Vectorization**: Embed windows using sentence-transformers models
4. **Scoring**: Calculate k-NN density scores (average distance to neighbors)
5. **Thresholding**: Select top X% based on score distribution
6. **Merging**: Combine overlapping significant windows
7. **Formatting**: Generate XML-tagged output with line references

### Scoring Methodology

- **Higher score** = Semantically unique = Anomalous
- **Lower score** = Repetitive = Normal background noise

The score for each window is the average cosine distance to its k nearest neighbors in the embedding space.

**Important:** This means repetitive patterns are filtered out even if they're critical errors. For example, the same FATAL error repeated 100 times will score as "normal" because it's semantically similar to itself. Cordon finds **unusual** patterns, not **all** errors.

**Want to understand the technical details?** Read the **[technical deep dive](./docs/how-it-works.md)** for the full algorithmic approach, mathematical foundations, and design decisions.

## Configuration Options

### Analysis Parameters

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `window_size` | 5 | `--window-size` | Number of lines per window |
| `stride` | 2 | `--stride` | Lines to skip between windows |
| `k_neighbors` | 5 | `--k-neighbors` | Number of neighbors for density calculation |
| `anomaly_percentile` | 0.1 | `--anomaly-percentile` | Percentile threshold (0.1 = top 10% most anomalous) |

### Backend Selection

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `backend` | `"sentence-transformers"` | `--backend` | Embedding backend (`sentence-transformers` or `llama-cpp`) |

### sentence-transformers Backend

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `model_name` | `"all-MiniLM-L6-v2"` | `--model-name` | HuggingFace model name |
| `batch_size` | 32 | `--batch-size` | Embedding batch size |
| `device` | `None` | `--device` | Device override (`cuda`/`mps`/`cpu`/`None` for auto) |
| `use_faiss` | `False` | `--use-faiss` | Force FAISS for k-NN (faster for large logs) |

### llama.cpp Backend

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `model_path` | `None` | `--model-path` | Path to GGUF model file (required for llama-cpp) |
| `n_gpu_layers` | 0 | `--n-gpu-layers` | Number of layers to offload to GPU (0=CPU only, -1=all) |
| `n_threads` | `None` | `--n-threads` | CPU threads for llama.cpp (`None`=auto-detect) |
| `n_ctx` | 2048 | `--n-ctx` | Context size for llama.cpp |

### Advanced Options

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `use_mmap_threshold` | 50000 | N/A | Auto-enable memory mapping above N windows (`None` to disable) |
| `use_faiss_threshold` | `None` | N/A | Auto-enable FAISS above N windows (requires faiss installation) |

Run `cordon --help` for full CLI documentation.

### ⚠️ Important: Token Limits and Window Sizing

**Transformer models have token limits that affect how much of each window is analyzed.** Windows exceeding the limit are automatically truncated to the first N tokens.

**Cordon will warn you if significant truncation is detected** and suggest better settings for your logs.

**Default model (`all-MiniLM-L6-v2`) has a 256-token limit:**
- Compact logs (20-30 tokens/line): Default `window_size=5` works perfectly
- Standard logs (40-50 tokens/line): Default settings work well
- Verbose logs (50-70 tokens/line): Consider larger window with a bigger model
- Very verbose logs (80+ tokens/line): Use a larger-context model

**For verbose system logs**, use larger-context models:
```bash
# BAAI/bge-base-en-v1.5 supports 512 tokens (~8-10 verbose lines)
cordon --model-name "BAAI/bge-base-en-v1.5" --window-size 8 --stride 4 your.log
```

**See [Configuration Guidelines](./docs/how-it-works.md#configuration-guidelines) for detailed recommendations.**

## Use Cases

### **What Cordon Is Good For:**

- **LLM Pre-processing**: Reduce large logs to small anomalous sections prior to analysis
- **Initial Triage**: First-pass screening of unfamiliar logs to find "what's unusual here?"
- **Anomaly Detection**: Surface semantically unique events (rare errors, state transitions, unusual clusters)
- **Exploratory Analysis**: Discover unexpected patterns without knowing what to search for

### **What Cordon Is NOT Good For:**

- **Complete error analysis** - Repetitive errors get filtered as "normal"
- **Specific error hunting** - Use `grep`/structured logging for known patterns
-
## Performance Considerations

Cordon automatically chooses the best approach based on log size:

| Strategy | When | RAM Usage | Speed | Notes |
|----------|------|-----------|-------|-------|
| **In-Memory** | <50k windows | ~200-500MB | Fastest | Default for typical logs |
| **Memory-Mapped** | 50k-500k windows | ~50-100MB | Moderate | Auto-enabled for large logs |
| **FAISS** | >500k windows | ~50MB | Fast | Requires optional `faiss` package |

**What's a "window"?** A window is a sliding chunk of N consecutive log lines (default: 10 lines). A 10,000-line log with window_size=10 and stride=5 creates ~2,000 windows.
