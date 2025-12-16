# Cordon Benchmark Suite

Evaluation framework for testing Cordon's semantic uniqueness detection on the HDFS dataset.

This benchmark measures:
- **Template Coverage**: Fraction of unique error types detected
- **Rare Template Detection**: Detection rate of rarest patterns
- **Frequency-Weighted Recall**: Inverse-frequency weighted metric
- **Traditional Metrics**: Line-level precision, recall, F1 for comparison
- **Model Performance**: Tests with BGE-large and MiniLM embedding models

See [METHODOLOGY.md](METHODOLOGY.md) for detailed methodology and [results/README.md](results/README.md) for benchmark results and analysis.

## Quick Start

### Full Benchmark (Evaluation + Visualizations + Structured Output)

```bash
# 1. Install dependencies
uv pip install -e ".[benchmark]"

# 2. Download HDFS dataset
python benchmark/download.py hdfs_v1

# 3. Run full benchmark (recommended)
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5 \
  --device cuda --batch-size 64 \
  --runs 5 --seed 42 \
  --generate-plots --output-dir benchmark/runs

# Output structure:
# benchmark/runs/run_<timestamp>/
#   ├── parameters.yaml          # Evaluation configuration
#   ├── results.json             # Detailed results
#   ├── aggregated_results.json  # Multi-run statistics
#   ├── plots/
#   │   ├── umap_100k.png              # UMAP projection
#   │   └── template_coverage_100k.png # Template detection chart
#   ├── data/                    # Reserved for future use
#   └── run_1/, run_2/, ...      # Individual run results (multi-run only)
```

### Quick Evaluation (Just Metrics, No Visualizations)

```bash
# Run evaluation only (prints to console)
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5
```

### Standalone Visualization (Advanced)

```bash
# Generate visualizations separately (if needed)
python benchmark/visualize.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 \
  --model BAAI/bge-large-en-v1.5
```

## Installation

### Prerequisites

- Python 3.10 or higher
- ~1GB free disk space (for HDFS dataset)
- 8GB+ RAM recommended

### Setup

```bash
# From project root
uv pip install -e ".[benchmark]"

# Verify installation
python -c "import cordon; import matplotlib; import yaml"
```

### Optional: GPU Acceleration

**For Embedding and Scoring (CUDA/MPS):**

Cordon now uses GPU acceleration for both embedding generation and k-NN scoring, providing significant speedups.

```bash
# NVIDIA GPUs (Pascal/GTX 10-series or newer)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Use with --device cuda flag
python benchmark/evaluate.py hdfs_v1 --device cuda --batch-size 64
```

**GPU Requirements:**
- NVIDIA: Compute capability >= 6.0 (GTX 1050+, RTX series, Tesla P/V/A/H series)
- Apple Silicon (M1/M2/M3): MPS support included, auto-detected

Cordon automatically detects and uses the best device. Both embedding AND scoring are GPU-accelerated.

**For Visualization (UMAP):**

UMAP visualizations can be accelerated with GPU support:

```bash
# NVIDIA GPUs with CUDA 11.x
uv pip install cuml-cu11

# NVIDIA GPUs with CUDA 12.x
uv pip install cuml-cu12
```

The code automatically detects and uses GPU-accelerated UMAP (cuML) if available, otherwise falls back to CPU (umap-learn).

## HDFS Dataset

| Metric | Value |
|--------|-------|
| **Total Lines** | 11.1M |
| **Sessions** | 575K |
| **Anomaly Rate** | 2.93% |
| **Size** | ~460MB |
| **Templates** | 29 unique event types |
| **Fail-Only Templates** | 12 templates only appear in failed sessions |

**Source:** [Loghub Repository](https://github.com/logpai/loghub) - "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics" (ISSRE 2023)

## Scripts

### 1. `evaluate.py` - Template-Level Evaluation (Primary Script)

**All-in-one script** for running evaluations with optional visualizations and structured output.

**Full Benchmark (Recommended):**
```bash
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5 \
  --runs 5 --seed 42 \
  --generate-plots --output-dir benchmark/runs
```

**Quick Evaluation (Console Only):**
```bash
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5
```

Measures semantic uniqueness detection using template coverage metrics.

**Output Options:**

1. **Console Only** (default): Prints metrics to terminal
2. **With Structured Output**: `--output-dir benchmark/runs` creates organized directories
3. **With Visualizations**: `--generate-plots` generates all plots
4. **Full Benchmark**: Combine both for complete workflow

**Recommended Full Benchmark:**
```bash
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5 \
  --runs 5 --seed 42 \
  --generate-plots --output-dir benchmark/runs --run-name my_experiment
```

**Output Structure:**

Single run:
```
benchmark/runs/my_experiment/
├── parameters.yaml
├── results.json
└── plots/
```

Multi-run:
```
benchmark/runs/my_experiment/
├── parameters.yaml
├── aggregated_results.json
├── plots/                     # Summary visualizations
└── run_1/, run_2/, .../       # Per-run results
    ├── results_run<N>.json
    └── plots/
```

**Core Options:**
- `--sample-size`: Number of lines (default: 100000, use 'full' for all 11M)
- `--window-size`: Window size (default: 4, non-overlapping windows)
- `--k-neighbors`: K-neighbors for density (default: 5, recommended: 10)
- `--anomaly-percentile`: Threshold percentile (default: 0.1, recommended: 0.02)
- `--model`: Embedding model (default: all-MiniLM-L6-v2, recommended: BAAI/bge-large-en-v1.5)

**Performance Options:**
- `--device`: Device for embedding and scoring (default: auto-detect, choices: cuda/mps/cpu)
- `--batch-size`: Embedding batch size (default: 32, increase for GPU)
- `--scoring-batch-size`: Scoring batch size (default: auto-detect based on GPU memory)

**Statistical Options:**
- `--runs`: Number of runs for statistics (default: 1)
- `--seed`: Random seed for reproducibility (default: random)

**Output Options:**
- `--generate-plots`: Generate all visualizations
- `--output-dir`: Create structured output directory
- `--run-name`: Custom name for run directory (default: timestamp)

See [`results/`](results/) for observed benchmark results.

### 2. Generated Visualizations

When using `--generate-plots`, the following visualizations are created:

**1. `umap_*.png` - UMAP Projection**
- Left panel: Colored by k-NN score (dark purple=low → pink/magenta=medium → orange/yellow=high)
- Right panel: Colored by ground truth (red=anomaly, green=normal)
- Shows embedding space structure and global relationships

**2. `template_coverage_*.png` - Template Detection Chart**
- Horizontal bar chart of all anomaly templates
- Green bars: Detected templates
- Red bars: Missed templates
- Sorted by frequency with log scale
- Compares detection results to ground truth

---

## Understanding the Visualizations

### UMAP Projection (`umap_*.png`)

**What it shows:** 2D projection of high-dimensional embedding space (384D → 2D).

**Left panel (colored by k-NN score):**
- Each point represents a log window
- Color (plasma scale): dark purple (low) → pink/magenta (medium) → orange/yellow (high)
- Spatial proximity in 2D approximates semantic similarity in 384D space
- Isolated points or small clusters with orange/yellow indicate detected anomalies
- Dense regions with dark purple indicate normal, repetitive patterns

**Right panel (colored by ground truth):**
- Same projection, colored by dataset's session-level labels
- Red points = ALL windows containing lines from sessions labeled "Anomaly" (includes both unique and repetitive patterns)
- Green points = windows from sessions labeled "Normal"
- Shows what the dataset considers anomalous (session-level), not what Cordon flagged
- Compares algorithm's scores (left) with dataset labels (right)

**How to read:**
- Orange/yellow on left + red on right = Cordon correctly scores windows from anomalous sessions as high
- Dark purple on left + red on right = Cordon correctly ignores repetitive errors in anomalous sessions (by design)
- Orange/yellow on left + green on right = Possible false positives OR unique patterns in normal sessions
- Pink/magenta points = medium-scoring windows (moderate semantic isolation)
- Isolated orange/yellow points indicate unique anomalies detected by semantic scoring
- Dense dark purple clusters indicate repetitive normal operations
- Note: Not all red points will be orange/yellow (many are repetitive errors from anomalous sessions)

### Template Coverage Chart (`template_coverage_*.png`)

**What it shows:** Detection success rate for each unique anomaly template type.

**Components:**
- Y-axis: template IDs (sorted by frequency, most common at top)
- X-axis: occurrence count on log scale (how many times each template appears)
- Green bars: templates that were detected in flagged lines
- Red bars: templates that were missed
- Stats box (top right): overall detection rate and rare template detection rate
- Legend box (bottom left): explanation of color coding

**How to read:**
- Longer bars = more common templates (many occurrences)
- Shorter bars = rarer templates (few occurrences)
- Green bars = successfully detected these anomaly types
- Red bars = missed these anomaly types
- Look for green in the shorter bars (detecting rare patterns is the goal)

**What this measures:** Diversity of anomaly types detected, weighted toward rarity.

**Key insight:** This chart directly measures semantic uniqueness detection. Finding 1 instance of a rare template (green bar at top) is more valuable than finding 1,000 instances of a common template (red bar at bottom).

---

### 2. `visualize.py` - Standalone Visualization (Optional)

**Note:** Visualizations are now integrated into `evaluate.py` via `--generate-plots`. This script is kept for backward compatibility and standalone use.

Generate visualizations separately (if not using `--generate-plots`):

```bash
python benchmark/visualize.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 \
  --model BAAI/bge-large-en-v1.5
```

**Saves to:** `benchmark/results/` (legacy location)

**Note:** Standalone `visualize.py` only generates UMAP projection. For template coverage visualization, use `evaluate.py` with `--generate-plots`.

---

### 3. `download.py` - Dataset Management

Download and extract the HDFS dataset.

```bash
# Download HDFS dataset (~460MB)
python benchmark/download.py hdfs_v1

# Force re-download
python benchmark/download.py hdfs_v1 --force
```

## Understanding the Metrics

**Template Recall**: Fraction of unique anomaly types detected (measures diversity, not quantity)

**Rare Template Recall**: Fraction of rare templates (< 100 occurrences) detected

**Frequency-Weighted Recall**: Inverse-frequency weighting (finding rare patterns = higher score)

**Traditional F1** (Precision/Recall): Line-level metrics for comparison. Expected to be low (0.05-0.15) for systems that ignore repetitive patterns.

See [METHODOLOGY.md](METHODOLOGY.md) for detailed metric definitions and evaluation process.

## Benchmark Results

### With Recommended Settings

**Configuration:**
```
window_size: 4 (non-overlapping), k_neighbors: 10
anomaly_percentile: 0.02, model: BAAI/bge-large-en-v1.5
device: cuda (GPU-accelerated embedding and scoring)
```

See [results/README.md](results/README.md) for complete benchmark results across multiple runs.

## Complete Evaluation Workflow

```bash
# 1. Download dataset
python benchmark/download.py hdfs_v1

# 2. Run full benchmark (evaluation + visualizations)
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5 \
  --runs 5 --seed 42 \
  --generate-plots --output-dir benchmark/runs \
  --run-name experiment_1

# 3. Review results
ls benchmark/runs/experiment_1/
# - parameters.yaml (configuration)
# - results.json (detailed results)
# - aggregated_results.json (multi-run statistics)
# - plots/ (all visualizations)
# - run_1/, run_2/, ... (individual runs)
```


## Dataset Source

- **HDFS v1**: [Loghub Repository](https://github.com/logpai/loghub)
- **Paper**: "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics" (ISSRE 2023)
- **Zenodo**: https://zenodo.org/records/8196385
