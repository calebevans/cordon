# Benchmark Results: Cordon on HDFS v1

**Dataset:** HDFS v1 from [Loghub](https://github.com/logpai/loghub) ([Zenodo: 8196385](https://zenodo.org/records/8196385))  
**Tests Conducted:** 15 configurations, 160 runs total  
**Evaluation Methodology:** See [METHODOLOGY.md](../METHODOLOGY.md)

---

## Executive Summary

**What This Benchmark Measures:**
- Ability to detect diverse types of anomalous log patterns (template recall)
- Emphasis on rare/unique pattern detection over common repetitive errors
- Stability and variance across different sample sizes and configurations

**Key Findings:**

1. **Sample size is the dominant factor**: Template recall improves from 58% (50k lines) to 97% (5M lines), while coefficient of variation decreases from 33% to 6% (at 1M lines).

2. **Model size has no impact**: BGE-large (1024-dim) and MiniLM (384-dim) perform identically across all sample sizes on HDFS logs.

3. **High variance at small sample sizes on HDFS logs**: Results on 50k-250k samples show CV > 15%, indicating instability and sensitivity to which specific portion of this repetitive, structured log file is sampled. Less repetitive log types may perform better at smaller sample sizes.

4. **Parameter tuning provides minimal benefit**: Adjusting window size and k-neighbors yields marginal improvements (~2%) that do not overcome fundamental small-sample limitations.

5. **Threshold changes affect traditional metrics only**: Varying anomaly percentile (0.02 → 0.10) impacts line-level precision/recall but leaves template-level metrics essentially unchanged.

**Critical Assessment:** Cordon demonstrates strong performance on large samples (≥500k lines) of HDFS logs with stable, high template recall. However, performance on small samples of this highly structured, repetitive dataset is highly variable. The system is best suited for large log files where comprehensive sampling is possible, though less repetitive log types (e.g., application logs with diverse error messages) may show better small-sample performance.

---

## Test Results Summary

### Baseline Tests (Standard Parameters)

| Sample Size | Config | Template Recall | Rare Recall | Freq-Weighted | Trad F1 | CV% |
|------------|---------|----------------|-------------|---------------|---------|-----|
| 50k | w5_k10_p02 | 58.3% ± 19.4% | 40.9% ± 31.5% | 0.325 ± 0.326 | 0.066 ± 0.037 | 33.2% |
| 100k | w5_k10_p02 | 66.8% ± 13.2% | 46.3% ± 31.5% | 0.339 ± 0.337 | 0.059 ± 0.029 | 19.7% |
| 250k | w5_k10_p02 | 76.7% ± 16.3% | 63.7% ± 31.1% | 0.529 ± 0.361 | 0.069 ± 0.032 | 21.3% |
| 500k | w5_k10_p02 | 84.0% ± 12.9% | 64.6% ± 33.4% | 0.525 ± 0.397 | 0.071 ± 0.040 | 15.4% |
| 1M | w5_k10_p02 | 93.7% ± 5.2% | 84.4% ± 14.8% | 0.731 ± 0.264 | 0.079 ± 0.029 | 5.6% |
| 5M | w5_k10_p02 | 96.6% | 90.0% | 0.507 | 0.080 | N/A |

**Key Observations:**
- Template recall increases from 58% to 97% as sample size grows
- Coefficient of variation (CV) decreases dramatically, indicating improved stability

### Parameter Tuning Tests (50k)

| Config | Template Recall | Rare Recall | Freq-Weighted | Trad F1 | CV% |
|--------|----------------|-------------|---------------|---------|-----|
| w5_k10_p02 (baseline) | 58.3% ± 19.4% | 40.9% ± 31.5% | 0.325 ± 0.326 | 0.066 ± 0.037 | 33.2% |
| w2_k5_p02 (tuned) | 60.1% ± 20.9% | 42.8% ± 33.5% | 0.353 ± 0.338 | 0.052 ± 0.027 | 34.8% |

**Key Observations:**
- Smaller windows (w2 vs w5) and lower k (k5 vs k10) provide marginal improvement (+1.8%)
- Variance remains high (CV > 30%) for both configurations at 50k sample size
- Tuning does not fundamentally solve small-sample instability on highly repetitive logs

### Threshold Sensitivity Tests

#### 50k Sample (w2_k5 tuned parameters)

| Threshold (p) | Template Recall | Rare Recall | Precision | Recall | CV% |
|--------------|----------------|-------------|-----------|--------|-----|
| 0.02 (2%) | 60.1% ± 20.9% | 42.8% ± 33.5% | 0.058 ± 0.034 | 0.057 ± 0.036 | 34.8% |
| 0.05 (5%) | 60.1% ± 20.9% | 42.8% ± 33.5% | 0.042 ± 0.024 | 0.096 ± 0.045 | 34.8% |
| 0.10 (10%) | 60.5% ± 20.8% | 43.4% ± 33.5% | 0.034 ± 0.019 | 0.149 ± 0.049 | 34.4% |

**Key Observations:**
- Template recall essentially unchanged across thresholds (60.1-60.5%)
- Higher thresholds increase traditional recall (0.057 → 0.149) at cost of precision (0.058 → 0.034)
- Threshold choice does not impact template-level metrics significantly

---

## Dataset Characteristics

### HDFS v1 Overview

**Source:** [Loghub HDFS v1](https://zenodo.org/records/8196385)  
**Domain:** Hadoop Distributed File System production logs  
**Size:** 11,175,629 lines (~1.5 GB)  
**Sessions:** 575,062 unique block IDs  
**Anomaly Rate:** 2.93% (16,838 anomalous sessions out of 575,062)  
**Templates:** 29 unique event types (E1-E29)

### Template Distribution

The HDFS dataset contains structured log messages that fit into 29 predefined templates. These templates represent different system events, from normal operations to various failure modes.

**Template Frequency Distribution:**
- **Common templates (>1000 occurrences):** E1, E3, E4, E5, E8 (normal operations)
- **Moderate templates (100-1000 occurrences):** E9, E11, E22, E23, E26 (mixed normal/anomaly)
- **Rare templates (<100 occurrences):** E2, E6, E13, E16, E18, E20, E25, E27, E28 (often critical failures)

**Anomalous Templates:**
Out of 29 templates, 17 appear in anomalous sessions in our samples. The distribution varies significantly:
- **Very rare anomalies:** E2 (1-2 occurrences), E28 (3-4 occurrences), E20 (4-5 occurrences)
- **Rare anomalies:** E6, E13, E16, E18, E25, E27 (<100 occurrences each)
- **Common anomalies:** E5, E22, E11, E9, E26 (>100 occurrences each)

### Why This Dataset is Challenging

HDFS logs represent highly structured, repetitive logging with only 29 template types across 11M+ lines. This low semantic diversity creates specific challenges:

**Semantic Similarity:** Many templates differ only in specific operations (e.g., "Adding block" vs "Deleting block"), making it difficult for embedding models to distinguish patterns.

**Repetitive Structure:** Anomalies are often variations of normal patterns (e.g., "Received block X" normally, but "Exception writing block X" is anomalous).

**Mixed-Context Templates:**
Several templates (E5, E9, E11, E22, E26) appear in both normal and anomalous sessions. These templates are not inherently anomalous; their anomaly status depends on context and co-occurrence with other events.

**Imbalanced Frequency:**
Rare templates (e.g., E2 with 1 occurrence) are extremely difficult to detect through density scoring, as they have no similar neighbors to establish a local density baseline.

**Low Semantic Diversity:**
With only 29 event types over 11M lines, HDFS has exceptionally low semantic diversity. Most real-world application logs have much higher diversity, which may lead to:
- Better small-sample coverage (each sample captures more variety)
- Lower sampling variance (less dependence on which specific portion is sampled)
- More stable anomaly detection at smaller sample sizes

### Ground Truth Limitations

**Session-Level Labels:**
Ground truth labels entire sessions (identified by block ID) as "Normal" or "Anomaly." This introduces imprecision:
- Not all lines in an anomalous session are necessarily anomalous
- An anomalous session may contain many normal operations
- Line-level metrics are affected by this mismatch

**Template Attribution:**
The evaluation uses line-level template matching: only templates that appear in the specific lines flagged by Cordon are credited as detected. This is more accurate than session-level attribution but still imperfect, as:
- A flagged line might be normal even in an anomalous session
- Template co-occurrence patterns matter more than individual templates
- No severity weighting is applied (all anomaly types treated equally)

**Sampling Variance:**
When sampling N lines from 11M total, the specific templates present vary significantly. Some rare templates may not appear in smaller samples at all, making "detection" impossible regardless of algorithm quality.

---

## Detailed Results Analysis

### Sample Size Impact

![Sample Size Scaling](analysis_graphs/sample_size_scaling.png)

**Template Recall vs Sample Size:**

| Sample Size | Template Recall | CV% |
|-------------|----------------|-----|
| 50k | 58.3% ± 19.4% | 33.2% |
| 100k | 66.8% ± 13.2% | 19.7% |
| 250k | 76.7% ± 16.3% | 21.3% |
| 500k | 84.0% ± 12.9% | 15.4% |
| 1M | 93.7% ± 5.2% | 5.6% |
| 5M | 96.6% | N/A |

**Key Observations:**

1. **Strong positive correlation:** Template recall increases monotonically with sample size. The relationship is roughly logarithmic, with diminishing returns at larger sizes.

2. **Variance decreases with sample size:** CV drops from 33% at 50k to 6% at 1M lines, indicating more consistent results with larger samples.

3. **Rare template detection improves:** Rare template recall increases from 41% (50k) to 84% (1M), showing even stronger improvement than overall template recall.

4. **Frequency-weighted recall trends higher:** The increase from 0.33 (50k) to 0.73 (1M) indicates the system increasingly favors rare patterns as sample size grows, which aligns with the design goal.

**Statistical Interpretation:**

The high variance at small sample sizes (CV 20-33%) indicates that results are highly dependent on *which* specific portion of the log file is sampled. With only 50k-250k lines from an 11M dataset of highly structured HDFS logs, random sampling may miss entire templates or capture non-representative distributions.

**Important Note:** This variance is likely exacerbated by HDFS logs' repetitive, template-heavy structure where only 29 event types exist. Application logs or other log types with greater semantic diversity per line may show better stability at smaller sample sizes, as each sample is more likely to contain diverse patterns.

At 1M lines (~9% of total dataset), sampling captures a more representative cross-section, leading to stable results across repeated runs with different random offsets.

**Inflection Point:**

The data suggests an inflection point around 500k lines where:
- Template recall reaches ~84% (diminishing returns beyond this)
- Variance becomes acceptable (CV ≈ 15%)
- Most unique template types are captured in samples

Beyond 500k, continued improvement follows a logarithmic curve with smaller marginal gains per additional line.

### Model Comparison

![Model Comparison](analysis_graphs/model_comparison.png)

**BGE-Large vs MiniLM Performance:**

| Sample Size | BGE-Large (w=5, 1024-dim) | MiniLM (w=4, 384-dim) | Difference |
|-------------|---------------------------|------------------------|------------|
| 50k | 58.3% ± 19.4% | 59.3% ± 19.7% | +1.0% |
| 100k | 66.8% ± 13.2% | 66.8% ± 13.7% | 0.0% |
| 250k | 76.7% ± 16.3% | 76.7% ± 16.3% | 0.0% |
| 500k | 84.0% ± 12.9% | 84.0% ± 12.9% | 0.0% |
| 1M | 93.7% ± 5.2% | 93.7% ± 5.2% | 0.0% |

**Key Finding:** Both models perform identically across all sample sizes on HDFS logs. The 384-dim all-MiniLM-L6-v2 and 1024-dim BAAI/bge-large-en-v1.5 show no meaningful difference in template detection performance.

**Interpretation:** For HDFS logs (29 templates, highly structured), semantic differences between templates are large and obvious (e.g., "error" vs "received"). Both model sizes easily capture these distinctions. Logs with greater semantic diversity or more subtle differences may show different model sensitivity.

### Parameter Tuning Impact

![Parameter Tuning Comparison](analysis_graphs/parameter_tuning_comparison.png)

Parameter tuning provided marginal improvements (~2%) that fall well within the variance of the baseline configuration. The overlapping confidence intervals (large standard deviations) indicate these improvements are not statistically significant.

Parameter tuning does not fundamentally solve the small-sample problem. The root cause of poor performance on 50k samples is insufficient data coverage, not suboptimal hyperparameters. Tuning may help at the margins but cannot overcome the fundamental constraint that 50k lines represents <0.5% of the full dataset.

**Conclusion:** For practical purposes, parameter tuning has negligible impact compared to simply using more data. Efforts to optimize hyperparameters for small samples yield limited returns.

### Variance and Stability Analysis

![Stability Box Plots](analysis_graphs/stability_boxplots.png)

**Coefficient of Variation Across Sample Sizes:**

| Sample Size | Mean Template Recall | Std Dev | CV% |
|-------------|---------------------|---------|-----|
| 50k | 58.3% | 19.4% | 33.2% |
| 100k | 66.8% | 13.2% | 19.7% |
| 250k | 76.7% | 16.3% | 21.3% |
| 500k | 84.0% | 12.9% | 15.4% |
| 1M | 93.7% | 5.2% | 5.6% |

**Box Plot Analysis:**

The box plots show the distribution of template recall across 10 runs for each sample size. Key features:
- **50k:** Wide distribution, large outliers, highly variable
- **100k-250k:** Moderate spread, some runs significantly better/worse than others
- **500k:** Narrowing distribution, approaching stability
- **1M:** Tight clustering around mean, minimal variance

**Individual Run Variability:**

At 50k, individual runs show template recall ranging from ~35% to ~85%, a 50-point spread. This means *which* 50k-line sample you get matters more than the algorithm itself.

At 1M, the range narrows to ~88% to ~98%, a 10-point spread. Results are now primarily algorithm-driven rather than sample-driven.

---

## Template-Level Analysis

### Consistently Detected Templates

**Templates detected in >90% of runs (1M sample):**
- E5, E22, E11, E9, E26, E21, E13, E7, E18, E16, E25, E27

These templates are reliably detected because they:
- Appear with sufficient frequency (typically >10 occurrences)
- Have distinct semantic content from normal operations
- Generate clear density anomalies in embedding space

### Consistently Missed Templates

**Templates missed in >50% of runs (across all sample sizes):**
- E20: "Unexpected error deleting block" (4-5 occurrences)
- E23: "delete: block is added to invalidSet" (50-100 occurrences)
- E28: "addStoredBlock: does not belong to any file" (3-4 occurrences)
- E6: "Received block of size X" (5-10 occurrences) - mixed normal/anomaly

**Why these templates are missed:**

1. **Extremely low frequency:** E2, E20, E28 appear 1-5 times in typical samples. With k-NN density scoring, these points may be isolated but their scores are **unstable due to lack of local context.**

2. **Semantic similarity to normal operations:** E6 and E23 describe operations that also occur normally, making them semantically close to dense "normal" clusters in embedding space.

3. **Sampling probability:** With 50k-250k samples from 11M lines, the probability of capturing a template with only 3-5 occurrences in the full dataset is low.

### Correlation with Template Frequency

**Detection Rate by Frequency Band:**

| Frequency | Templates in Band | Avg Detection Rate (1M) |
|-----------|------------------|------------------------|
| 1-10 occurrences | E2, E6, E16, E18, E20, E25, E28 | 71% |
| 11-50 occurrences | E13, E27 | 95% |
| 51-100 occurrences | E7, E21, E22, E23 | 88% |
| >100 occurrences | E5, E9, E11, E26 | 100% |

**Observation:** Detection rate correlates positively with frequency, but the relationship is not linear. The 1-10 occurrence band shows high variability (50-100% detection rates) depending on semantic distinctiveness.

**Implication:** Cordon is effective at detecting moderately rare events (10-100 occurrences) but struggles with extremely rare events (<10 occurrences), especially if they are semantically similar to normal operations.

---

## Traditional Metrics Context

![Traditional vs Template Metrics](analysis_graphs/traditional_vs_template_metrics.png)

Traditional line-level F1 scores (observed: 5-8%) are low by design. Cordon ignores repetitive patterns and optimizes for diversity of types detected, not total instance coverage. Template-level metrics better reflect this approach by measuring unique pattern types found rather than counting all anomalous lines.

---

## Configuration Details

### Common Parameters

**BGE-Large Tests (w5_k10_p02, w2_k5, etc.):**
- **Model:** BAAI/bge-large-en-v1.5 (512 token context, 1024-dim embeddings)
- **Window Size:** 5 (baseline), 2-4 (tuned variants)
- **Batch Size:** 512 (embedding generation)

**MiniLM Tests (w4_k10_p02_minilm):**
- **Model:** all-MiniLM-L6-v2 (256 token context, 384-dim embeddings)
- **Window Size:** 4
- **Batch Size:** 32 (embedding generation)

**All Tests:**
- **Device:** CUDA (GPU acceleration)
- **Seed:** 42 (reproducible sampling)
- **Runs:** 10 per configuration

### Variable Parameters

**Baseline (w5_k10_p02):**
- window_size: 5
- k_neighbors: 10
- anomaly_percentile: 0.02 (top 2%)

**50k Tuned (w2_k5):**
- window_size: 2 (smaller for limited data)
- k_neighbors: 5 (lower density requirement)
- anomaly_percentile: 0.02/0.05/0.10 (threshold variants)

**100k Tuned (w3_k7):**
- window_size: 3
- k_neighbors: 7
- anomaly_percentile: 0.05

**250k Tuned (w4_k8):**
- window_size: 4
- k_neighbors: 8
- anomaly_percentile: 0.04

---

## Metrics Explanation

### Template-Level Metrics (Primary)

**Template Recall:** Fraction of unique anomaly template types detected
- **Interpretation:** Measures diversity of anomaly patterns found
- **Goal:** High recall indicates comprehensive coverage of anomaly types

**Rare Template Recall:** Fraction of rare templates (< 100 occurrences) detected
- **Interpretation:** Ability to find infrequent but potentially critical patterns
- **Goal:** High recall shows effectiveness at detecting novel events

**Frequency-Weighted Recall:** Inverse-frequency weighted score (rare = higher value)
- **Interpretation:** Emphasizes finding rare patterns over common ones
- **Goal:** Higher scores indicate bias toward unique pattern detection

### Traditional Metrics (Reference)

**Line-Level F1, Precision, Recall:** Standard anomaly detection metrics
- **Note:** Expected to be low (0.05-0.08)
- **Reason:** Cordon ignores repetitive patterns, traditional metrics count all instances
- **Use case:** Reference only, not primary evaluation criteria

### Stability Metrics

**Coefficient of Variation (CV%):** (std / mean) × 100
- **< 15%:** Stable results
- **> 15%:** High sensitivity to sampling, results may vary
- **Implication:** Small samples (50k-250k) of HDFS show high variance

---

## Limitations and Considerations

### Dataset Limitations

**Single Dataset:**
All results are based solely on HDFS v1 logs. Generalization to other log types (application logs, web server logs, system logs) is unvalidated. HDFS logs are highly structured and repetitive with only 29 template types over 11M lines; logs with greater semantic diversity per line may show:
- Better performance on smaller sample sizes (more variability captured per sample)
- Lower variance (less dependent on which specific sample is drawn)
- Different rare event detection characteristics

**Template Dependency:**
Evaluation requires pre-parsed templates. For datasets without template ground truth, this methodology cannot be applied without first performing template extraction (e.g., using Drain or similar parsers).

**Session-Level Ground Truth:**
Ground truth labels entire sessions, not individual lines. This introduces noise in line-level metrics, though template-level evaluation is less affected.

**Limited Anomaly Diversity:**
HDFS contains 17 anomalous template types. Datasets with hundreds of anomaly types may show different behavior, particularly regarding rare pattern detection.

### Evaluation Scope Limitations

**No Temporal Metrics:**
This benchmark does not measure:
- Detection latency (time from anomaly occurrence to detection)
- Temporal ordering (whether anomalies are detected in sequence)
- Streaming performance (assumes batch processing)

**No Severity Weighting:**
All anomaly templates are treated equally. In practice, some anomalies are more critical than others. This benchmark does not account for severity or business impact.

**Rare Template Threshold Arbitrary:**
The 100-occurrence threshold for "rare" templates is pragmatic but not theoretically justified. Different thresholds would change rare detection metrics.

**Sampling Variance:**
Results are based on random samples with seed=42. Different seeds would produce slightly different results, especially at small sample sizes.

### Methodology Limitations

**Percentile Thresholding:**
Using anomaly_percentile (e.g., top 2%) is adaptive but may flag near-identical content as "anomalous" in very uniform logs. Absolute threshold methods may be more appropriate for some use cases.

**k-NN Density Assumptions:**
The approach assumes semantic isolation (low local density) indicates anomaly. This fails for:
- Normal but rare operations (e.g., system startup sequences)
- Clustered anomalies (multiple similar errors in same region)

**Embedding Model Dependency:**
Results depend on the quality of semantic embeddings. Different models may produce different results. No ablation study was performed to quantify this dependency.

### Applicability Constraints

**This benchmark is appropriate for:**
- Systems prioritizing semantic uniqueness detection
- Log summarization and context reduction for LLM analysis
- Exploratory analysis to find diverse error patterns
- Large log files (>500k lines) with sufficient data

**This benchmark is NOT appropriate for:**
- Real-time monitoring systems (requires batch processing)
- Complete error coverage requirements (ignores repetitive patterns)
- Compliance logging (misses many instances of known errors)
- Small log files (<100k lines with high variance on repetitive logs)

---

## Conclusions

### Performance Summary

**What Cordon Does Well:**
1. **High template recall on large samples:** Achieves 97% recall of unique anomaly types at 5M lines
2. **Bias toward rare patterns:** Frequency-weighted recall indicates preference for detecting rare/unique events
3. **Stable results at scale:** CV drops to 6% at 1M lines
4. **Effective rare template detection:** 90% recall of rare templates at 5M sample

**What Cordon Does Not Do Well (on HDFS logs):**
1. **Variable on small samples of repetitive logs:** CV >20% for samples <250k lines of HDFS; results highly variable on this structured, template-heavy dataset. Less repetitive log types may perform better.
2. **Misses extremely rare events:** Templates with 1-5 occurrences often missed (~71% detection rate)
3. **No instance coverage:** Traditional F1 of 5-8% indicates most instances of known errors are ignored
4. **Sensitive to sampling on sparse datasets:** At small sizes with limited template diversity (29 types in HDFS), which specific lines are sampled matters significantly

### Recommended Use Cases

**Use Cordon When:**
- You have large log files (>500k lines recommended for highly repetitive logs)
- Working with logs that have semantic diversity (application logs, diverse error messages)
- Goal is to identify diverse types of issues, not count every instance
- Reducing log volume for LLM analysis or human review
- Exploring unfamiliar logs to discover what kinds of errors exist

**Exercise Caution When:**
- Working with small samples (<100k lines) of highly repetitive, template-heavy logs (like HDFS)
- Logs have limited semantic diversity (e.g., only 29 event types over millions of lines)
- Results may be more stable on smaller samples if logs have high semantic variability per line

**Do NOT Use Cordon When:**
- Need to detect every instance of specific known errors
- Require real-time detection
- Compliance requires complete coverage

### Comparison to Alternatives

**vs Traditional Anomaly Detection (frequency counting, regex):**
- Cordon: Better at finding novel/rare patterns
- Traditional: Better at complete coverage of known patterns

**vs LLM Direct Analysis:**
- Cordon: Reduces log volume by 84-98% depending on threshold (98% with p=0.02 on HDFS, 85% with p=0.1 on diverse logs)
- LLM Direct: Overwhelmed by repetitive content in large logs

**vs Template Extraction (Drain, etc.):**
- Cordon: Identifies unusual patterns via semantic similarity
- Template Extraction: Parses log structure into templates

---

## Directory Structure

```
results/
├── README.md                       (this file)
├── analysis_graphs/                (comparison visualizations)
│   ├── sample_size_scaling.png
│   ├── parameter_tuning_comparison.png
│   ├── stability_boxplots.png
│   ├── threshold_sensitivity.png
│   ├── traditional_vs_template_metrics.png
│   └── model_comparison.png
├── generate_analysis_graphs.py     (script to regenerate graphs)
├── 50k/                            (50,000 line tests)
│   ├── 50k_w5_k10_p02_bge/        (BGE-large: w=5, k=10, p=0.02)
│   ├── 50k_w4_k10_p02_minilm/     (MiniLM: w=4, k=10, p=0.02)
│   ├── 50k_w2_k5_p02_bge/         (BGE-large: tuned w=2, k=5, p=0.02)
│   ├── 50k_w2_k5_p05_bge/         (BGE-large: tuned w=2, k=5, p=0.05)
│   └── 50k_w2_k5_p10_bge/         (BGE-large: tuned w=2, k=5, p=0.10)
├── 100k/                           (100,000 line tests)
│   ├── 100k_w5_k10_p02_bge/       (BGE-large: w=5, k=10, p=0.02)
│   ├── 100k_w4_k10_p02_minilm/    (MiniLM: w=4, k=10, p=0.02)
│   └── 100k_w3_k7_p05_bge/        (BGE-large: tuned w=3, k=7, p=0.05)
├── 250k/                           (250,000 line tests)
│   ├── 250k_w5_k10_p02_bge/       (BGE-large: w=5, k=10, p=0.02)
│   ├── 250k_w4_k10_p02_minilm/    (MiniLM: w=4, k=10, p=0.02)
│   └── 250k_w4_k8_p04_bge/        (BGE-large: tuned w=4, k=8, p=0.04)
├── 500k/                           (500,000 line tests)
│   ├── 500k_w5_k10_p02_bge/       (BGE-large: w=5, k=10, p=0.02)
│   └── 500k_w4_k10_p02_minilm/    (MiniLM: w=4, k=10, p=0.02)
├── 1M/                             (1,000,000 line tests)
│   ├── 1M_w5_k10_p02_bge/         (BGE-large: w=5, k=10, p=0.02)
│   └── 1M_w4_k10_p02_minilm/      (MiniLM: w=4, k=10, p=0.02)
└── 5M/                             (5,000,000 line tests)
    └── 5M_w5_k10_p02_bge/         (BGE-large: w=5, k=10, p=0.02)
```

---

## Reproduction

To reproduce any test:

```bash
# BGE-large example (100k)
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 5 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model BAAI/bge-large-en-v1.5 \
  --device cuda --batch-size 512 \
  --runs 10 --seed 42 \
  --generate-plots \
  --output-dir benchmark/results/100k/100k_w5_k10_p02_bge

# MiniLM example (100k)
python benchmark/evaluate.py hdfs_v1 \
  --sample-size 100000 \
  --window-size 4 \
  --k-neighbors 10 --anomaly-percentile 0.02 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda --batch-size 32 \
  --runs 10 --seed 42 \
  --generate-plots \
  --output-dir benchmark/results/100k/100k_w4_k10_p02_minilm
```

See [METHODOLOGY.md](../METHODOLOGY.md) for complete evaluation methodology.

---

## Appendix: Detailed Results Tables

### A.1 BGE-Large Results (w5_k10_p02)

| Sample | Template Recall | Rare Recall | Freq-Weighted | Trad F1 | CV% | Runs |
|--------|----------------|-------------|---------------|---------|-----|------|
| 50k | 0.583 ± 0.194 | 0.409 ± 0.315 | 0.325 ± 0.326 | 0.066 ± 0.037 | 33.2% | 10 |
| 100k | 0.668 ± 0.132 | 0.463 ± 0.315 | 0.339 ± 0.337 | 0.059 ± 0.029 | 19.7% | 10 |
| 250k | 0.767 ± 0.163 | 0.637 ± 0.311 | 0.529 ± 0.361 | 0.069 ± 0.032 | 21.3% | 10 |
| 500k | 0.840 ± 0.129 | 0.646 ± 0.334 | 0.525 ± 0.397 | 0.071 ± 0.040 | 15.4% | 10 |
| 1M | 0.937 ± 0.052 | 0.844 ± 0.148 | 0.731 ± 0.264 | 0.079 ± 0.029 | 5.6% | 10 |
| 5M | 0.966 | 0.900 | 0.507 | 0.080 | N/A | 1 |

### A.2 MiniLM Results (w4_k10_p02)

| Sample | Template Recall | Rare Recall | Freq-Weighted | Trad F1 | CV% | Runs |
|--------|----------------|-------------|---------------|---------|-----|------|
| 50k | 0.593 ± 0.197 | 0.422 ± 0.322 | 0.338 ± 0.326 | 0.073 ± 0.035 | 33.2% | 10 |
| 100k | 0.668 ± 0.137 | 0.471 ± 0.323 | 0.341 ± 0.338 | 0.073 ± 0.029 | 20.5% | 10 |
| 250k | 0.767 ± 0.163 | 0.637 ± 0.311 | 0.529 ± 0.361 | 0.080 ± 0.036 | 21.3% | 10 |
| 500k | 0.840 ± 0.129 | 0.646 ± 0.334 | 0.525 ± 0.397 | 0.081 ± 0.046 | 15.4% | 10 |
| 1M | 0.937 ± 0.052 | 0.844 ± 0.148 | 0.731 ± 0.264 | 0.088 ± 0.031 | 5.6% | 10 |

### A.3 Parameter Tuning Results (50k, BGE-Large)

| Config | W | K | P | Template Recall | CV% |
|--------|---|---|---|----------------|-----|
| Baseline | 5 | 10 | 0.02 | 0.583 ± 0.194 | 33.2% |
| Tuned | 2 | 5 | 0.02 | 0.601 ± 0.209 | 34.8% |
| Tuned + p05 | 2 | 5 | 0.05 | 0.601 ± 0.209 | 34.8% |
| Tuned + p10 | 2 | 5 | 0.10 | 0.605 ± 0.208 | 34.4% |

### A.4 Test Configuration Summary

**Total Tests Conducted:** 15 configurations  
**Total Runs:** 160 individual runs  
**Total Compute Time:** ~72 hours (GPU: NVIDIA CUDA)  
**Total Lines Analyzed:** ~85 million (across all runs)  
**Seeds Used:** 42 (for reproducibility)  
**Models Used:**
- BAAI/bge-large-en-v1.5 (1024-dim embeddings, 512 token context) - 10 configurations
- all-MiniLM-L6-v2 (384-dim embeddings, 256 token context) - 5 configurations