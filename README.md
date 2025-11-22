# Cordon

**Semantic anomaly detection for system log files**

Cordon is a Python library that uses transformer-based embeddings and density-based scoring to identify **semantically unusual patterns** in large log files. It's designed to reduce massive logs down to the most anomalous sections for analysis by LLMs or human operators.

**Key principle:** Repetitive patterns (even errors) are considered "normal background." Cordon surfaces **unusual, rare, or clustered events** that stand out semantically from the bulk of the logs.

## Features

- **Semantic Analysis**: Uses transformer models to understand log content meaning, not just keyword matching
- **Density-Based Scoring**: Identifies anomalies using k-NN distance in embedding space
- **Noise Reduction**: Filters out repetitive logs, keeping only unusual patterns

## Installation

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

For very large logs (optional FAISS support for better performance):

```bash
# For CPU-only systems
uv pip install -e ".[faiss-cpu]"

# For systems with CUDA/GPU
uv pip install -e ".[faiss-gpu]"
```

## Quick Start

### Command Line

```bash
# Basic usage
cordon system.log

# Multiple files
cordon app.log error.log

# With options
cordon --window-size 20 --k-neighbors 10 --anomaly-percentile 0.05 application.log

# Detailed statistics
cordon --detailed --device cuda production.log
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
    window_size=20,          # Lines per window
    stride=10,               # Step size for sliding
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

| Parameter | Default | CLI Flag | Description |
|-----------|---------|----------|-------------|
| `window_size` | 10 | `--window-size` | Number of lines per window |
| `stride` | 5 | `--stride` | Lines to skip between windows |
| `k_neighbors` | 5 | `--k-neighbors` | Number of neighbors for density calculation |
| `anomaly_percentile` | 0.1 | `--anomaly-percentile` | Percentile threshold (0.1 = top 10% most anomalous) |
| `model_name` | `"all-MiniLM-L6-v2"` | `--model-name` | Sentence-transformers model |
| `batch_size` | 32 | `--batch-size` | Embedding batch size |
| `device` | `None` | `--device` | Device override (`cuda`/`mps`/`cpu`/`None` for auto) |
| `use_mmap_threshold` | 50000 | N/A | Auto-enable memory mapping above N windows (`None` to disable) |
| `use_faiss_threshold` | `None` | N/A | Auto-enable FAISS above N windows (requires faiss installation) |

Run `cordon --help` for full CLI documentation.

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
