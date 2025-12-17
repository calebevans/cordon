# Cordon

[![PyPI version](https://img.shields.io/pypi/v/cordon.svg)](https://pypi.org/project/cordon/)
[![License](https://img.shields.io/github/license/calebevans/cordon.svg)](https://github.com/calebevans/cordon/blob/main/LICENSE)

**Semantic anomaly detection for system log files**

Cordon uses transformer embeddings and k-NN density scoring to identify semantically unusual patterns in large log files, reducing massive logs to the most anomalous sections for analysis.

**Key principle:** Repetitive patterns (even errors) are considered "normal background." Cordon surfaces unusual, rare, or clustered events that stand out semantically from the bulk of the logs.

> 📖 For an in-depth explanation of the methodology, see my Red Hat Developer article: [Semantic anomaly detection in log files with Cordon](https://developers.redhat.com/articles/2025/12/09/semantic-anomaly-detection-log-files-cordon).

## Features

- **Semantic Analysis**: Uses transformer models to understand log content meaning, not just keyword matching
- **k-NN Density Scoring**: Identifies anomalies using k-NN distance in embedding space
- **Noise Reduction**: Filters out repetitive logs, keeping only unusual patterns
- **Multiple Backends**: sentence-transformers (default), llama.cpp for containers, or remote APIs (OpenAI, Gemini, etc.)

## Requirements

### GPU Requirements (Optional but Recommended)

For GPU acceleration, you need:
- **NVIDIA GPU**: Pascal architecture or newer (GTX 10-series, RTX series, Tesla P/V/A/H series)
- **Compute Capability**: 6.0 or higher
- **Compatible GPUs**: GTX 1050+, RTX 20/30/40 series, Tesla P100+, V100, A100, H100

**Not compatible**: GTX 900-series or older (Maxwell/Kepler architectures)

CPU mode is always available as a fallback.

## Installation

### From PyPI (Recommended)

```bash
# With uv (recommended)
uv pip install cordon

# With pip
pip install cordon
```

### From Source

```bash
# Clone the repository
git clone https://github.com/calebevans/cordon.git
cd cordon

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

### Container Installation

```bash
make container-build
```

See [Container Guide](./docs/CONTAINER.md) for GPU support and advanced usage.

## Quick Start

### Command Line

```bash
# Basic usage
cordon system.log

# Multiple files
cordon app.log error.log

# With options
cordon --window-size 10 --k-neighbors 10 --anomaly-percentile 0.05 app.log

# Filter to a percentile range (exclude top 5%, keep next 10%)
cordon --anomaly-range 0.05 0.15 app.log

# With GPU acceleration (scoring batch size auto-detected)
cordon --device cuda --batch-size 64 large.log

# Override auto-detection if needed
cordon --device cuda --batch-size 64 --scoring-batch-size 50000 large.log

# Save results to file
cordon --output anomalies.xml system.log

# Show detailed statistics and save results
cordon --detailed --output results.xml app.log

# llama.cpp backend (for containers)
cordon --backend llama-cpp system.log
```

> **Note:** On first run, Cordon downloads the embedding model. Subsequent runs use the cached model.

### Python Library

```python
from pathlib import Path
from cordon import SemanticLogAnalyzer, AnalysisConfig

# Basic usage
analyzer = SemanticLogAnalyzer()
output = analyzer.analyze_file(Path("system.log"))
print(output)

# Advanced configuration with GPU acceleration
config = AnalysisConfig(
    window_size=10,
    k_neighbors=10,
    anomaly_percentile=0.05,
    device="cuda",           # GPU for embedding and scoring
    batch_size=64,           # Embedding batch size
    scoring_batch_size=None  # Auto-detect optimal batch size (default)
)
analyzer = SemanticLogAnalyzer(config)
result = analyzer.analyze_file_detailed(Path("app.log"))

# Using anomaly range mode (exclude top 5%, keep next 10%)
config = AnalysisConfig(
    window_size=10,
    k_neighbors=10,
    anomaly_range_min=0.05,  # Exclude top 5% (most extreme anomalies)
    anomaly_range_max=0.15,  # Include up to 15% (keep next 10%)
    device="cuda",
)
analyzer = SemanticLogAnalyzer(config)
result = analyzer.analyze_file_detailed(Path("app.log"))

# Using remote API backend
import os
config = AnalysisConfig(
    backend="remote",
    model_name="openai/text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    batch_size=100,
)
analyzer = SemanticLogAnalyzer(config)
result = analyzer.analyze_file_detailed(Path("app.log"))
```

## Example Output

Running `cordon examples/apache_sample.log` produces output like this (showing one representative block from the full output):

```
================================================================================
Analyzing: examples/apache_sample.log
Total lines: 2,004
================================================================================

<block lines="581-600" score="0.1746">
[Sun Dec 04 07:18:00 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 07:18:00 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 07:18:00 2005] [error] mod_jk child workerEnv in error state 7
[Sun Dec 04 07:45:45 2005] [error] [client 63.13.186.196] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 08:54:17 2005] [error] [client 147.31.138.75] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 09:35:12 2005] [error] [client 207.203.80.15] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 10:53:30 2005] [error] [client 218.76.139.20] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 11:11:07 2005] [error] [client 24.147.151.74] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 11:33:18 2005] [error] [client 211.141.93.88] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 11:42:43 2005] [error] [client 216.127.124.16] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 12:33:13 2005] [error] [client 208.51.151.210] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 13:32:32 2005] [error] [client 65.68.235.27] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 14:29:00 2005] [error] [client 4.245.93.87] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 15:18:36 2005] [error] [client 67.154.58.130] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 15:59:01 2005] [error] [client 24.83.37.136] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 16:24:03 2005] [notice] jk2_init() Found child 1219 in scoreboard slot 6
[Sun Dec 04 16:24:05 2005] [error] [client 58.225.62.140] Directory index forbidden by rule: /var/www/html/
[Sun Dec 04 16:24:06 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
[Sun Dec 04 16:24:06 2005] [error] mod_jk child workerEnv in error state 6
[Sun Dec 04 16:31:07 2005] [notice] jk2_init() Found child 1248 in scoreboard slot 7
</block>

... additional anomalous blocks ...
```

The tool identified this block as semantically unusual (score: 0.1746) because it contains a cluster of client-specific directory access errors from various IPs—a different pattern from the repetitive worker initialization messages that dominate the rest of the log file. The full output contains multiple such blocks, representing the most anomalous windows based on the configured threshold (default: top 10%).

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
cordon --backend llama-cpp system.log

# With GPU acceleration
cordon --backend llama-cpp --n-gpu-layers 10 system.log

# Custom model
cordon --backend llama-cpp --model-path ./model.gguf system.log
```

See [llama.cpp Guide](./docs/llama-cpp.md) for details on models, performance, and GPU setup.

### Remote API Backend

Use remote embedding APIs from OpenAI, Gemini, Cohere, and more via LiteLLM.

```bash
# OpenAI (uses OPENAI_API_KEY env var, or pass --api-key)
cordon --backend remote \
  --model-name openai/text-embedding-3-small \
  system.log

# Gemini (uses GEMINI_API_KEY env var)
cordon --backend remote \
  --model-name gemini/text-embedding-004 \
  system.log

# Cohere (uses COHERE_API_KEY env var)
cordon --backend remote \
  --model-name cohere/embed-english-v3.0 \
  system.log

# Or explicitly pass API key
cordon --backend remote \
  --model-name openai/text-embedding-3-small \
  --api-key $OPENAI_API_KEY \
  system.log

# Custom endpoint (OpenAI-compatible)
cordon --backend remote \
  --model-name text-embedding-3-small \
  --endpoint http://localhost:8000/v1 \
  system.log
```

**Supported providers** (via LiteLLM): OpenAI, Azure OpenAI, Gemini, Cohere, Bedrock, Voyage AI, Mistral, Hugging Face, and any OpenAI-compatible endpoint.

## Container Usage

### Build

```bash
# Build locally
make container-build
```

### Run

```bash
# Pull published image from GitHub Container Registry
podman pull ghcr.io/calebevans/cordon:latest  # or :dev for development builds

# Run with published image
podman run --rm -v /path/to/logs:/logs:Z ghcr.io/calebevans/cordon:latest /logs/system.log

# Run with locally built image
make container-run DIR=/path/to/logs ARGS="/logs/system.log"

# With GPU (requires Podman with libkrun)
podman run --device /dev/dri -v /path/to/logs:/logs:Z ghcr.io/calebevans/cordon:latest \
  --backend llama-cpp --n-gpu-layers 10 /logs/system.log
```

See [Container Guide](./docs/CONTAINER.md) for full details.

## Primary Use Case: LLM Context Reduction

Cordon attempts to solve the problem of log files being too large for LLM context windows by reducing them to semantically significant sections.

**Real-world reduction rates from benchmarks:**
- 1M-line HDFS logs → 20K lines (98% reduction with p=0.02 threshold)
- 5M-line HDFS logs → 100K lines (98% reduction with p=0.02 threshold)

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
2. **Segmentation**: Create non-overlapping windows of N lines
3. **Vectorization**: Embed windows using transformer models
4. **Scoring**: Calculate k-NN density scores
5. **Thresholding**: Select top X% based on scores
6. **Merging**: Combine adjacent significant windows
7. **Formatting**: Generate XML-tagged output

### Scoring

- **Higher score** = Semantically unique = Anomalous
- **Lower score** = Repetitive = Normal background noise

The score for each window is the average cosine distance to its k nearest neighbors in the embedding space.

**GPU Acceleration**: Both embedding and scoring phases automatically leverage GPU acceleration (CUDA/MPS) when available, providing significant speedups for large log files.

**Important:** Repetitive patterns are filtered even if critical. The same FATAL error repeated 100 times scores as "normal" because it's semantically similar to itself.

See [Cordon's architecture](./docs/architecture.md) for full details.

## Configuration

### Analysis Parameters

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `window_size` | 4 | `--window-size` | Lines per window (non-overlapping) |
| `k_neighbors` | 5 | `--k-neighbors` | Number of neighbors for density calculation |
| `anomaly_percentile` | 0.1 | `--anomaly-percentile` | Top N% to keep (0.1 = 10%) |
| `anomaly_range_min` | None | `--anomaly-range MIN MAX` | Lower bound for range mode (exclude top X%) |
| `anomaly_range_max` | None | `--anomaly-range MIN MAX` | Upper bound for range mode (include up to Y%) |
| `batch_size` | 32 | `--batch-size` | Batch size for embedding generation |
| `scoring_batch_size` | Auto | `--scoring-batch-size` | Batch size for k-NN scoring (auto-detects based on GPU memory) |

**Note on Filtering Modes:**
- **Percentile Mode** (default): `--anomaly-percentile 0.1` keeps the top 10% most anomalous windows
- **Range Mode**: `--anomaly-range 0.05 0.15` excludes the top 5% (most extreme) and keeps the next 10% (moderately anomalous). Useful for filtering out known issues or startup noise while focusing on unusual-but-not-extreme patterns.

### Backend Options

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `backend` | `sentence-transformers` | `--backend` | Embedding backend (sentence-transformers/llama-cpp/remote) |
| `model_name` | `all-MiniLM-L6-v2` | `--model-name` | Model name (HuggingFace for sentence-transformers, provider/model for remote) |
| `device` | Auto | `--device` | Device for embedding and scoring (cuda/mps/cpu) |
| `model_path` | None | `--model-path` | GGUF model path (llama-cpp) |
| `n_gpu_layers` | 0 | `--n-gpu-layers` | GPU layers (llama-cpp) |
| `api_key` | None | `--api-key` | API key for remote embeddings (falls back to env vars) |
| `endpoint` | None | `--endpoint` | Custom API endpoint URL (remote) |
| `request_timeout` | 60.0 | N/A | Request timeout in seconds (remote) |

### Output Options

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `detailed` | False | `--detailed` | Show detailed statistics (timing, score distribution) |
| `output` | None | `--output`, `-o` | Save anomalous blocks to file (default: stdout) |

Run `cordon --help` for full CLI documentation.

### ⚠️ Important: Token Limits and Window Sizing

**Transformer models have token limits that affect how much of each window is analyzed.** Windows exceeding the limit are automatically truncated to the first N tokens.

**Cordon will warn you if significant truncation is detected** and suggest better settings for your logs.

**Default model (`all-MiniLM-L6-v2`) has a 256-token limit:**
- Compact logs (20-30 tokens/line): Can increase to `window_size=8` for more context
- Standard logs (40-50 tokens/line): Default works well
- Verbose logs (50-70 tokens/line): Default works, or use larger model for bigger windows
- Very verbose logs (80+ tokens/line): Reduce to `window_size=3` or use larger-context model

**For verbose system logs**, use larger-context models:
```bash
# BAAI/bge-base-en-v1.5 supports 512 tokens (~8-10 verbose lines)
cordon --model-name "BAAI/bge-base-en-v1.5" --window-size 8 your.log
```

**See [Configuration Guidelines](./docs/architecture.md#configuration-guidelines) for detailed recommendations.**

## Use Cases

### What Cordon Is Good For

- **LLM Pre-processing**: Reduce large logs to small anomalous sections prior to analysis
- **Initial Triage**: First-pass screening of unfamiliar logs to find "what's unusual here?"
- **Anomaly Detection**: Surface semantically unique events (rare errors, state transitions, unusual clusters)
- **Exploratory Analysis**: Discover unexpected patterns without knowing what to search for
- **Iterative Investigation**: Use range mode to exclude known issues and focus on the next tier of anomalies

### What Cordon Is NOT Good For

- Complete error analysis (repetitive errors filtered)
- Specific error hunting (use grep/structured logging)
- Compliance logging (this is lossy by design)

### Choosing Between Percentile and Range Mode

**Use Percentile Mode** (`--anomaly-percentile`) when:
- First time analyzing a log file
- You want the most anomalous content
- Simple, straightforward filtering

**Use Range Mode** (`--anomaly-range`) when:
- You want to exclude known extreme anomalies (startup errors, expected failures)
- Investigating the "next tier" of unusual patterns
- The top percentile is dominated by things like start-up logs
- You need to focus on moderately anomalous patterns

**Example workflow:**
```bash
# Step 1: Find the most extreme anomalies
cordon --anomaly-percentile 0.05 app.log > top5.xml

# Step 2: After reviewing, exclude those and see the next tier
cordon --anomaly-range 0.05 0.15 app.log > next10.xml

# Step 3: Focus on moderate anomalies, excluding startup noise
cordon --anomaly-range 0.02 0.10 app.log > filtered.xml
```

## Performance

### GPU Acceleration

Cordon automatically leverages GPU acceleration for both embedding and scoring phases when available:

- **Embedding**: Uses PyTorch/sentence-transformers with CUDA or MPS
- **Scoring**: Uses PyTorch for GPU-accelerated k-NN computation
- **Speedup**: 5-15x faster scoring on GPU compared to CPU for large datasets

For large log files (millions of lines), GPU acceleration can reduce total processing time from hours to minutes.

### Memory Management

Cordon uses PyTorch for all k-NN scoring operations:

| Strategy | When | RAM Usage | Speed |
|----------|------|-----------|-------|
| PyTorch GPU | GPU available (CUDA/MPS) | Moderate | Fastest |
| PyTorch CPU | No GPU / CPU forced | Moderate | Fast |

**What's a "window"?** A window is a non-overlapping chunk of N consecutive log lines (default: 4 lines). A 10,000-line log with window_size=4 creates 2,500 windows.
