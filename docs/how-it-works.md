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
- **Cost scaling**: More tokens = exponentially higher costs
- **Signal-to-noise**: LLMs waste tokens on repetitive content, burying important information

**Example**: Filling GPT-4's 128K context with raw logs would technically work, but the model's ability to extract insights diminishes significantly when processing mostly-repetitive content at that scale.

**Cordon's solution**: Significantly reduce logs while keeping semantically interesting content. (In our tests: 84.7% reduction on average across diverse log types)

---

## Technical Approach

Cordon uses a **density-based anomaly detection** approach in **semantic embedding space**.

### High-Level Algorithm

```
1. Chunk logs into overlapping windows (e.g., 10 lines per window)
2. Embed each window using a transformer model → 384-dimensional vectors
3. For each embedding, compute k-NN distance to find "neighbors"
4. Score = average distance to k nearest neighbors
5. Higher score = farther from neighbors = more anomalous
6. Keep top X% highest-scoring windows
7. Merge overlapping windows into contiguous blocks
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
# Configuration
window_size = 10  # Lines per window
stride = 5        # Lines to advance between windows

# Example
Log lines:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Window 1:   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Window 2:         [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Window 3:                     [11, 12, 13, 14, 15, ...]
```

**Why overlapping windows?**
- **Boundary handling**: Anomalies at window edges appear in multiple windows
- **Context preservation**: Each window has sufficient surrounding context
- **Robust detection**: Reduces sensitivity to window alignment

**Trade-off**: More windows = more computation, but better accuracy.

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

### 3. Anomaly Scoring

**Uses k-NN distance in embedding space to quantify "unusualness".**

```python
k_neighbors = 5  # Number of nearest neighbors to consider

For each window embedding e_i:
    1. Find k nearest neighbors: {e_j1, e_j2, ..., e_jk}
    2. Compute cosine distances: d(e_i, e_j1), d(e_i, e_j2), ...
    3. Score = mean(distances)
```

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

**Selects top anomalies using percentile-based threshold.**

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

### 5. Merging

**Combines overlapping significant windows into contiguous blocks.**

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
- **Cleaner output**: One block instead of overlapping fragments

**Algorithm**: Interval merging with score tracking (similar to interval scheduling).

### 6. Output Formatting

**Generates structured XML output with metadata.**

```xml
<block lines="145-158" score="0.2341">
[Original log content here]
</block>
```

**Metadata included:**
- **Line range**: References back to original file
- **Anomaly score**: Quantifies unusualness
- **Raw content**: Preserves original formatting

---

## Mathematical Foundation

### Embedding Space Geometry

**Vector space properties:**
```
Vectors: ℝ^384 (384-dimensional real-valued vectors)
Distance metric: Cosine distance = 1 - cosine_similarity
Similarity: cos(θ) = (a·b) / (||a|| ||b||)
```

**Why cosine distance?**
- **Scale-invariant**: Focuses on direction, not magnitude
- **Natural for text**: "Connection timeout" × 3 vs. × 1 have same direction
- **Clustering-friendly**: Forms well-separated clusters
- **Range [0, 2]**: Easy to interpret (0 = identical, 1 = orthogonal, 2 = opposite)

### k-NN Density Estimation

**Theoretical basis:** Local Outlier Factor (LOF) family of algorithms.

```
Density at point p ≈ k / volume_of_k_ball(p)

Where:
- k = number of neighbors
- volume_of_k_ball(p) = average distance to k nearest neighbors
```

**Intuition:**
- **High density** → Small k-ball contains many points → Normal
- **Low density** → Large k-ball needed to find k points → Anomalous

**Cordon's simplification:**
- Uses average k-NN distance directly (proxy for inverse density)
- Faster than full LOF computation
- Sufficient for log analysis (doesn't need precise density ratios)

### Why k=5 Default?

Trade-offs in choosing k:

| k value | Pros | Cons |
|---------|------|------|
| k=1 | Detects isolated outliers | Sensitive to noise |
| k=5 | Good balance | May miss subtle patterns |
| k=20 | Robust to noise | May miss small anomaly clusters |

**Empirical finding**: k=5 provides good balance in practice for typical log files.

---

## Design Decisions & Trade-offs

### 1. Window Size (Default: 10 lines)

**Why 10 lines?**
- **Sufficient context**: Typical log events often span multiple lines
- **Semantic coherence**: Related log lines cluster together
- **Computational efficiency**: Not too many windows

**Trade-offs:**
- **Smaller windows** (5 lines): Faster, but less context, more noise
- **Larger windows** (20 lines): Better context, but mixes events, slower

**When to adjust:**
- **Verbose logs** (stack traces): Increase to 20-30
- **Compact logs** (single-line events): Decrease to 5
- **Very large files**: Increase to reduce window count

### 2. Stride (Default: 5 lines = 50% overlap)

**Why 50% overlap?**
- **Redundancy**: Ensures anomalies appear in ≥2 windows
- **Boundary robustness**: Events split across windows still detected
- **Reasonable cost**: ~2× window count vs. non-overlapping (good trade-off)

**Alternative approaches:**
- **No overlap** (stride=window_size): Faster, but misses boundary events
- **High overlap** (stride=2, 80% overlap): More robust, but ~2.4× more windows than default

### 3. Model Choice: all-MiniLM-L6-v2

**Why this model?**
- **Fast**: Efficient inference on CPU
- **Compact**: 384 dimensions
- **Accurate**: Strong performance on semantic similarity tasks
- **Pre-trained**: Works out-of-box for log text

**Alternatives considered:**
- **all-mpnet-base-v2**: More accurate (768-dim), but slower
- **all-MiniLM-L12-v2**: Slightly better, but slower
- **Domain-specific models**: Better for specialized logs, but require training

**Trade-off**: Accuracy vs. speed. MiniLM-L6-v2 is the sweet spot for general logs.

### 4. Percentile Threshold (Default: 10%)

**Why top 10%?**
- **Empirical balance**: Captures interesting events without overwhelming output
- **Adaptive**: Works across different log types (security, system, application)
- **LLM-friendly**: Reduces 2,000 lines → ~306 lines average (based on our test results)

**Adjustment guidance:**
- **5%**: Maximum compression (only most unusual events)
- **10%**: Good default (balanced signal-to-noise)
- **20%**: Comprehensive view (includes moderately unusual patterns)

---

## Performance Optimizations

### Memory Management

**Challenge**: Large logs create many windows → high memory usage.

**Solutions:**
1. **Lazy loading**: Read log lines on-demand
2. **Streaming embeddings**: Process in batches, don't store all at once
3. **Memory-mapped arrays**: Store embeddings on disk for huge logs
4. **FAISS indexing**: Approximate k-NN for millions of windows

**Automatic scaling:**
```python
< 50K windows:  In-memory NumPy arrays (fastest)
≥ 50K windows:  Memory-mapped arrays (RAM-efficient, auto-enabled)
Optional:       FAISS approximate search (for very large logs, must be installed)
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

---

**Last Updated**: November 22, 2025
**Version**: 0.1.0
