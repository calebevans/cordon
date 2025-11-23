# Cordon

**Semantic anomaly detection for system log files**

Cordon uses transformer-based embeddings and density-based scoring to identify semantically unusual patterns in large log files, designed to reduce massive logs down to the most anomalous sections for analysis.

**Key principle:** Repetitive patterns (even errors) are considered "normal background." Cordon surfaces unusual, rare, or clustered events that stand out semantically from the bulk of the logs.

## Features

- **Semantic Analysis**: Uses transformer models to understand log content meaning, not just keyword matching
- **Density-Based Scoring**: Identifies anomalies using k-NN distance in embedding space
- **Noise Reduction**: Filters out repetitive logs, keeping only unusual patterns
- **Multiple Backends**: sentence-transformers (default) or llama.cpp for containers

## Installation

### Native Installation

```bash
# With uv (recommended)
uv pip install -e .

# With pip
pip install -e .
```

For development:

```bash
uv pip install -e ".[dev]"
pre-commit install
```

For llama.cpp backend (GPU acceleration in containers):

```bash
uv pip install -e ".[llama-cpp]"
```

For FAISS support (better performance on large logs):

```bash
uv pip install -e ".[faiss-cpu]"  # CPU
uv pip install -e ".[faiss-gpu]"  # GPU
```

### Container Installation

```bash
make container-build
```

See [Container Guide](./docs/container.md) for GPU support and advanced usage.

## Quick Start

### Command Line

```bash
# Basic usage
cordon system.log

# Multiple files
cordon app.log error.log

# With options
cordon --window-size 10 --k-neighbors 10 --anomaly-percentile 0.05 app.log

# With FAISS for large logs
cordon --use-faiss large.log

# llama.cpp backend (for containers)
cordon --backend llama-cpp --use-faiss system.log
```

### Python Library

```python
from pathlib import Path
from cordon import SemanticLogAnalyzer, AnalysisConfig

# Basic usage
analyzer = SemanticLogAnalyzer()
output = analyzer.analyze_file(Path("system.log"))
print(output)

# Advanced configuration
config = AnalysisConfig(
    window_size=10,
    stride=5,
    k_neighbors=10,
    anomaly_percentile=0.05,
    device="cuda"
)
analyzer = SemanticLogAnalyzer(config)
result = analyzer.analyze_file_detailed(Path("app.log"))
```

## Backend Options

### sentence-transformers (Default)

Best for native installations with GPU access.

```bash
cordon system.log  # Auto-detects GPU (MPS/CUDA)
cordon --device cuda system.log
cordon --device cpu system.log
```

### llama.cpp Backend

Best for container deployments with GPU acceleration via Vulkan.

```bash
# Auto-downloads model on first run
cordon --backend llama-cpp --use-faiss system.log

# With GPU acceleration
cordon --backend llama-cpp --use-faiss --n-gpu-layers 10 system.log

# Custom model
cordon --backend llama-cpp --use-faiss --model-path ./model.gguf system.log
```

See [llama.cpp Guide](./docs/llama-cpp.md) for details on models, performance, and GPU setup.

## Container Usage

```bash
# Build
make container-build

# Run
make container-run ARGS="system.log"

# With GPU (requires Podman with libkrun)
podman run --device /dev/dri -v $(pwd)/logs:/logs cordon:latest \
  --backend llama-cpp --use-faiss --n-gpu-layers 10 /logs/system.log
```

See [Container Guide](./docs/container.md) for full details.

## Primary Use Case: LLM Context Reduction

Cordon attempts to solve for when log files are too large for a context window:

```
Problem: 50GB production log → Can't fit in any LLM context window
Solution: Cordon → 12 anomalous blocks (few KB) → Send to LLM for analysis
```

Example workflow:

```python
# Extract anomalies
analyzer = SemanticLogAnalyzer()
anomalies = analyzer.analyze_file(Path("production.log"))

# Send curated context to LLM (now fits in context window)
```

The output is intentionally lossy—it discards repetitive patterns to focus on semantically unusual events.

## How It Works

### Pipeline

1. **Ingestion**: Read log file line-by-line
2. **Segmentation**: Create overlapping windows of N lines
3. **Vectorization**: Embed windows using transformer models
4. **Scoring**: Calculate k-NN density scores
5. **Thresholding**: Select top X% based on scores
6. **Merging**: Combine overlapping significant windows
7. **Formatting**: Generate XML-tagged output

### Scoring

- **Higher score** = Semantically unique = Anomalous
- **Lower score** = Repetitive = Normal background noise

The score for each window is the average cosine distance to its k nearest neighbors in the embedding space.

**Important:** Repetitive patterns are filtered even if critical. The same FATAL error repeated 100 times scores as "normal" because it's semantically similar to itself.

See [Cordon's architecture](./docs/architecture.md) for full details.

## Configuration

### Analysis Parameters

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `window_size` | 5 | `--window-size` | Lines per window |
| `stride` | 2 | `--stride` | Lines to skip between windows |
| `k_neighbors` | 5 | `--k-neighbors` | Number of neighbors for density calculation |
| `anomaly_percentile` | 0.1 | `--anomaly-percentile` | Top N% to keep (0.1 = 10%) |

### Backend Options

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `backend` | `sentence-transformers` | `--backend` | Embedding backend |
| `model_name` | `all-MiniLM-L6-v2` | `--model-name` | HuggingFace model |
| `device` | Auto | `--device` | Device (cuda/mps/cpu) |
| `model_path` | None | `--model-path` | GGUF model path (llama-cpp) |
| `n_gpu_layers` | 0 | `--n-gpu-layers` | GPU layers (llama-cpp) |
| `use_faiss` | False | `--use-faiss` | Use FAISS for large logs |

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

**See [Configuration Guidelines](./docs/architecture.md#configuration-guidelines) for detailed recommendations.**

## Use Cases

### What Cordon Is Good For

- **LLM Pre-processing**: Reduce large logs to small anomalous sections prior to analysis
- **Initial Triage**: First-pass screening of unfamiliar logs to find "what's unusual here?"
- **Anomaly Detection**: Surface semantically unique events (rare errors, state transitions, unusual clusters)
- **Exploratory Analysis**: Discover unexpected patterns without knowing what to search for

### What Cordon Is NOT Good For

- Complete error analysis (repetitive errors filtered)
- Specific error hunting (use grep/structured logging)
- Compliance logging (this is lossy by design)

## Performance

Cordon automatically chooses the best approach:

| Strategy | When | RAM Usage | Speed |
|----------|------|-----------|-------|
| In-Memory | <50k windows | ~200-500MB | Fastest |
| Memory-Mapped | 50k-500k windows | ~50-100MB | Moderate |
| FAISS | >500k windows | ~50MB | Fast |

**What's a "window"?** A window is a sliding chunk of N consecutive log lines (default: 10 lines). A 10,000-line log with window_size=10 and stride=5 creates ~2,000 windows.

See [Test Examples](./docs/examples/) for real-world results across 9 log types.
