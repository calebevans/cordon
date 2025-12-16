# How Cordon Works: Technical Deep Dive

This document explains the approach and technical methodology behind Cordon's semantic anomaly detection system.

## Table of Contents

1. [Core Concept](#core-concept)
2. [The Problem](#the-problem-space)
3. [Technical Approach](#technical-approach)
4. [Pipeline Stages](#pipeline-stages)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
7. [Performance Optimizations](#performance-optimizations)

---

## Core Concept

**Cordon detects anomalies based on semantic uniqueness, not pattern matching or frequency analysis.**

The fundamental insight: In log files, **repetitive patterns are normal** (even if they're errors), while **semantically unusual patterns are interesting**. A critical error that appears once is more anomalous than the same error repeated 1,000 times.

---

## The Problem

### Log Files Are Highly Repetitive

Real-world log files typically exhibit:
- **Highly repetitive content**: Normal operations repeated many times
- **Temporal patterns**: Same sequences recurring on schedules
- **Structured templates**: Similar messages with varying parameters

Example:
```
[2024-01-01 10:00:01] INFO: Processing request #12345  <- Repeated many times
[2024-01-01 10:00:02] INFO: Processing request #12346
[2024-01-01 10:00:03] INFO: Processing request #12347
...
[2024-01-01 15:23:42] ERROR: OutOfMemoryError: Java heap space  <- Appears once!
```

Traditional tools would focus on the ERROR keyword, but Cordon identifies the ERROR as anomalous because it's **semantically different** from the surrounding context.

### The LLM Context Window Problem

Modern LLM-driven log analysis faces:
- **Context limits**: GPT-4 has ~128K token context (≈100K log lines)
- **Quality degradation**: Even when logs fit, performance degrades as context fills up (especially with repetitive content)
- **Cost scaling**: More tokens = higher costs
- **Signal-to-noise**: LLMs waste tokens on repetitive content, burying important information

**Example**: Filling GPT-4's 128K context with raw logs would technically work, but the model's ability to extract insights diminishes significantly when processing mostly-repetitive content at that scale.

**Cordon's solution**: Reduce logs while keeping semantically interesting content. (Benchmark: 98% reduction on 1M-5M line HDFS logs with p=0.02 threshold)

---

## Technical Approach

Cordon uses **density scoring for anomaly detection** in **semantic embedding space**.

### High-Level Algorithm

```
1. Chunk logs into non-overlapping windows (e.g., 10 lines per window)
2. Embed each window using a transformer model → 384-dimensional vectors
3. For each embedding, compute k-NN distance to find "neighbors"
4. Score = average distance to k nearest neighbors
5. Higher score = farther from neighbors = more anomalous
6. Keep top X% highest-scoring windows
7. Merge adjacent windows into contiguous blocks
```

### Why This Works

**Semantic embeddings cluster similar content:**
- Normal operations → Dense clusters (many neighbors nearby)
- Anomalous events → Sparse regions (few/no neighbors)
- Different error types → Separate regions

**k-NN distance measures isolation:**
- Low distance = Many similar logs nearby = Normal
- High distance = No similar logs nearby = Anomalous

---

## Pipeline Stages

### 1. Ingestion & Windowing

**Windowing converts variable-length logs into fixed-size semantic units.**

```python
# Configuration (defaults)
window_size = 4  # Lines per window

# Example
Log lines:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
Window 1:   [1, 2, 3, 4]
Window 2:           [5, 6, 7, 8]
Window 3:                   [9, 10, 11, 12]
Window 4:                           [13, 14, 15, 16]
```

**Non-overlapping windows provide:**
- **Clear boundaries**: Each log line appears in exactly one window
- **Efficient processing**: Fewer windows to analyze
- **Better anomaly isolation**: Anomalies are not normalized across overlapping windows

### 2. Semantic Embedding

**Transforms text windows into dense vector representations.**

```python
Model: "all-MiniLM-L6-v2" (sentence-transformers)
Input:  "ERROR: Connection timeout\nRetrying...\nERROR: Failed"
Output: [0.23, -0.45, 0.12, ..., 0.67]  # 384-dimensional vector
```

**Why sentence-transformers?**
- **Semantic understanding**: Trained on paraphrase detection
- **Fast inference**: Efficient on CPU, even faster on GPU
- **Reasonable size**: 384 dimensions balances expressiveness vs. memory
- **Pre-trained**: No domain-specific training needed

**Key property**: Semantically similar text → nearby vectors
```
"Connection timeout" ≈ "Request timed out" ≈ "No response from server"
"OutOfMemoryError"   ≠ "Connection timeout"
```

#### Token Limit Constraints

**Important**: Transformer models have maximum token limits that affect how much of each window is actually analyzed.

**Token limits by model:**
```
all-MiniLM-L6-v2:      256 tokens (default, 384-dim embeddings)
all-mpnet-base-v2:     384 tokens (768-dim embeddings)
BAAI/bge-base-en-v1.5: 512 tokens (768-dim embeddings)
BAAI/bge-large-en:     512 tokens (1024-dim embeddings)
```

**When a window exceeds the token limit, the transformer automatically truncates to the first N tokens.** The rest of the window content is silently ignored during embedding.

**Example with verbose system logs** (typical: 50-60 tokens per line):
```
window_size=4:  ~200-240 tokens → fits in 256 limit → all lines analyzed ✓ (default)
window_size=5:  ~250-300 tokens → fits in 256 limit → most lines analyzed ✓
window_size=10: ~500-600 tokens → exceeds 256 limit → only first ~4 lines analyzed
window_size=50: ~2,500-3,000 tokens → exceeds 256 limit → only first ~4 lines analyzed
```

**This means:** With the default model and verbose logs, large window sizes provide diminishing returns because only the beginning of each window is embedded.

**Cordon automatically detects truncation** and warns you with recommendations for better settings.

**Recommendations:**
1. **Match window_size to token limits**: For 50-60 token/line logs, use `window_size=4` with `all-MiniLM-L6-v2`
2. **Use larger-context models**: Switch to `BAAI/bge-base-en-v1.5` for 512-token windows (~8 lines)
3. **Check your logs**: Run a sample through a tokenizer to estimate tokens per line

**Trade-off**: Larger context models are slower to encode but provide better semantic understanding of longer windows.

### 3. Anomaly Scoring

**Uses k-NN distance in embedding space to quantify "unusualness".**

```python
k_neighbors = 5  # Number of nearest neighbors to consider

For each window embedding e_i:
    1. Find k nearest neighbors: {e_j1, e_j2, ..., e_jk}
    2. Compute cosine distances: d(e_i, e_j1), d(e_i, e_j2), ...
    3. Score = mean(distances)
```

**GPU-Accelerated Implementation:**

Cordon uses PyTorch for k-NN scoring, providing significant speedups via GPU acceleration:

```python
# Scoring with PyTorch (GPU or CPU)
scoring_batch_size = 10000  # Process 10k windows at once

For each batch of embeddings:
    1. Compute pairwise cosine similarities: similarity = batch @ embeddings.T
    2. Convert to distances: distance = 1 - similarity
    3. Find k nearest: torch.topk(distance, k=k_neighbors, largest=False)
    4. Average distances (excluding self)
```

**Performance benefits:**
- **GPU acceleration**: 5-15x faster than CPU-only sklearn for large datasets
- **Matrix operations**: Highly optimized PyTorch kernels
- **Batch processing**: Configurable via `--scoring-batch-size` for memory/speed tuning
- **Device flexibility**: Automatically uses CUDA, MPS, or CPU based on availability

**Scoring intuition:**
- **Dense cluster** (normal logs): Many neighbors close → Low average distance → Low score
- **Sparse region** (anomaly): Nearest neighbors far away → High average distance → High score
- **Isolated point** (unique event): No close neighbors → Very high score

**Example scores:**
```
Score 0.01: "INFO: Request processed" (very common, repeated frequently)
Score 0.05: "WARN: Cache miss" (moderately common)
Score 0.30: "FATAL: Database corruption detected" (rare, semantically unique)
```

### 4. Thresholding

**Selects anomalies using percentile or range filtering.**

#### Percentile Mode (Default)

```python
anomaly_percentile = 0.1  # Keep top 10% most anomalous

1. Compute all window scores: [0.01, 0.02, 0.03, ..., 0.35]
2. Calculate 90th percentile: threshold = 0.12
3. Select windows where score > threshold
```

**Why percentile vs. absolute threshold?**
- **Adaptive**: Works across different log types without tuning
- **Relative**: Finds "most unusual" in any dataset
- **Robust**: Not sensitive to score scale variations

**Trade-off**: Very uniform logs might flag near-identical content as "anomalous."

#### Range Mode (Advanced)

**Filters for anomalies within a specific percentile band, excluding the most extreme.**

```python
anomaly_range_min = 0.05  # Exclude top 5% (most extreme)
anomaly_range_max = 0.15  # Include up to 15% (keep next 10%)

1. Compute all window scores: [0.01, 0.02, 0.03, ..., 0.35]
2. Calculate upper threshold (95th percentile): upper = 0.30
3. Calculate lower threshold (85th percentile): lower = 0.20
4. Select windows where lower <= score < upper
```

**When to use range mode:**
- **Filter startup noise**: Exclude the most extreme anomalies that might be initialization issues
- **Focus on moderate anomalies**: Find unusual patterns that aren't outliers
- **Known issues**: Exclude top X% if you know certain errors are expected but want to see what else is unusual
- **Iterative analysis**: After reviewing top anomalies, look at the next tier

**Example use case**:
```bash
# First pass: see top 5% most anomalous
cordon --anomaly-percentile 0.05 app.log > top5.xml

# Second pass: exclude those, see next 10%
cordon --anomaly-range 0.05 0.15 app.log > next10.xml
```

### 5. Merging

**Combines adjacent significant windows into contiguous blocks.**

```python
Significant windows (line ranges):
  Window A: lines 10-20 (score=0.15)
  Window B: lines 15-25 (score=0.18)
  Window C: lines 20-30 (score=0.12)

Merged block: lines 10-30 (score=max(0.15, 0.18, 0.12) = 0.18)
```

**Why merge?**
- **Reduces redundancy**: Same anomaly shouldn't appear 5 times
- **Preserves context**: Extended anomalous sequences stay together
- **Cleaner output**: One block instead of separate fragments

**Algorithm**: Interval merging with score tracking (similar to interval scheduling).

### 6. Output Formatting

**Generates structured XML output with metadata.**

```xml
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
```

**Metadata included:**
- **Line range**: References back to original file
- **Anomaly score**: Quantifies unusualness
- **Raw content**: Preserves original formatting

---

## Performance Optimizations

### Memory Management

**Challenge**: Large logs create many windows → high memory usage.

**Solutions:**
1. **Lazy loading**: Read log lines on-demand
2. **Streaming embeddings**: Process in batches, don't store all at once
3. **Memory-mapped arrays**: Store embeddings on disk for huge logs

**Automatic scaling:**
```python
< 50K windows:  In-memory NumPy arrays (fastest)
≥ 50K windows:  Memory-mapped arrays (RAM-efficient, auto-enabled)
```
### Batching Strategy

**Embedding batches:**
```python
batch_size = 32  # Process 32 windows at once

Benefits:
- GPU utilization: Amortize kernel launch overhead
- CPU cache: Better memory locality
- SIMD: Vectorized operations

Trade-off: Larger batches use more VRAM but are faster
```

### k-NN Scoring

**Challenge**: k-NN distance calculations can be slow for large datasets.

**Solution**: PyTorch implementation leveraging GPU or CPU.

```python
# Device selection (automatic)
device = "cuda"  # or "mps" or "cpu" (auto-detected)
scoring_batch_size = 10000  # Tune based on memory

# Scoring uses PyTorch for matrix operations:
# - CUDA (NVIDIA GPUs): Fastest, highly optimized
# - MPS (Apple Silicon): Fast on M1/M2/M3
# - CPU: Uses optimized PyTorch matrix operations

# Override via CLI:
cordon --device cuda --scoring-batch-size 50000 large.log  # High memory GPU
cordon --device cpu --scoring-batch-size 10000 system.log  # CPU
```

**Performance characteristics:**
- **GPU (CUDA/MPS)**: Best for large datasets (>100k windows)
  - Leverages parallel matrix operations
  - 5-15x faster than CPU
  - Higher batch sizes possible with GPU memory
- **CPU (PyTorch)**: Good for all dataset sizes
  - Optimized matrix operations
  - Efficient memory usage
  - Used when no GPU available

**Batch size tuning:**
- **Small GPU memory**: `--scoring-batch-size 10000` (conservative)
- **Large GPU memory**: `--scoring-batch-size 50000` (faster, uses more VRAM)
- **CPU**: `--scoring-batch-size 10000` (default works well)

### Hardware Acceleration

**Device auto-detection:**
```python
Priority order:
1. CUDA (NVIDIA GPUs) - fastest for both phases
2. MPS (Apple Silicon) - fast on M1/M2/M3
3. CPU - slowest but universal

# The --device flag controls both embedding and scoring
cordon --device cuda system.log  # GPU for both phases
cordon --device cpu system.log   # CPU for both phases

# Auto-detection (default):
# - Tries CUDA first, then MPS, then falls back to CPU
```

**Performance impact:**
- **Embedding**: GPU provides 5-10x speedup for transformer inference
- **Scoring**: GPU provides 5-15x speedup for k-NN calculations
- **Combined**: Large log files process in minutes instead of hours on GPU

---

## Configuration Guidelines

### Choosing Window Size

**The relationship between window_size and token limits is critical for effective analysis.**

**Step 1: Measure your log verbosity**
```bash
# Sample your log and count average tokens per line
head -n 100 your.log | wc -c  # Rough estimate: chars ≈ 1.3x tokens
```

**Step 2: Choose window size based on model**
```
Token budget per window = model_token_limit
Lines per window ≈ token_limit / tokens_per_line

Example (50 token/line logs):
- all-MiniLM-L6-v2 (256 tokens):    window_size ≤ 5
- all-mpnet-base-v2 (384 tokens):   window_size ≤ 7
- BAAI/bge-base-en-v1.5 (512 tokens): window_size ≤ 10
```

**Common configurations:**

| Log Type | Tokens/Line | Recommended Config |
|----------|-------------|-------------------|
| **Compact** (app logs) | 20-30 | Increase to `window_size=8` for more context |
| **Standard** (web server) | 40-50 | Default `window_size=4` ✓ |
| **Verbose** (system logs) | 50-70 | Default `window_size=4` ✓ or use larger model |
| **Very verbose** (debug logs) | 80+ | Reduce to `window_size=3` or use larger model |

---

**Last Updated**: November 22, 2025
**Version**: 0.1.0
