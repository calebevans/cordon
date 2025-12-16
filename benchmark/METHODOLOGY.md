# Benchmark Methodology: Template-Level Evaluation for Semantic Anomaly Detection

**Dataset:** HDFS v1 from Loghub

This evaluation methodology measures semantic anomaly detection systems that prioritize **uniqueness over frequency**. Traditional line-level metrics (Precision, Recall, F1) don't work well for systems designed to ignore repetitive patterns. Instead, this uses **template-level coverage metrics** that measure the diversity of anomaly types detected rather than the quantity of instances flagged.

The approach uses line-level template attribution (only credits templates in flagged lines).

## Motivation

### The Problem with Traditional Metrics

Traditional anomaly detection benchmarks use line-level metrics (Precision, Recall, F1) where each correctly identified anomalous line counts as a "True Positive."

This doesn't work for semantic uniqueness detection because:
- Finding 1 of 10,000 identical errors yields 0.01% recall
- Common errors dominate the metric
- Systems are rewarded for flagging repetitive patterns

Finding all 10,000 instances of a repetitive error scores better than finding 5 instances of a rare critical error, even though the rare error is more valuable to identify.

### What Semantic Uniqueness Detection Does

Cordon uses embedding density estimation to identify patterns that are **semantically isolated** in log space. It assigns high scores to rare, unusual patterns and low scores to repetitive patterns (even if labeled as errors).

This is a different task than traditional anomaly detection, so it needs different evaluation metrics that measure diversity of types detected rather than completeness of instance coverage.

---

## Methodology

### Template-Level Evaluation

Instead of counting line instances, count **unique anomaly types** (templates) detected.

A template is a generalized pattern representing a class of log messages. For example, template E20 represents: `"Unexpected error trying to delete block [*] BlockInfo not found in volumeMap [*]"` where `[*]` represents variable content.

**Metrics:**

**Template Recall**: Fraction of unique anomaly templates detected
```
Template Recall = |Detected Templates ∩ Anomaly Templates| / |Anomaly Templates|
```

**Rare Template Recall**: Fraction of rare templates detected (< 100 occurrences)

**Frequency-Weighted Recall**: Inverse-frequency weighting (finding rare templates = higher value)
```
Weight(template) = 1 / occurrence_count
FW Recall = Σ weights(detected) / Σ weights(all anomalies)
```

### Line-Level Template Attribution

For each flagged log line:
1. Parse the line
2. Match it to a template using regex patterns
3. Credit the template only if it appears in the flagged line itself

This ensures we only credit templates the system actually flagged, not templates from the same session.

---

## Dataset: HDFS v1

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Source** | [Loghub](https://github.com/logpai/loghub) ([Zenodo: 8196385](https://zenodo.org/records/8196385)) |
| **Total Lines** | 11.1 million |
| **Sessions** | 575,062 |
| **Anomaly Rate** | 2.93% (16,838 anomalous sessions) |
| **Templates** | 29 unique event types (E1-E29) |
| **Fail-Only Templates** | 12 templates appear exclusively in failed sessions |

### Ground Truth

**Session-Level Labels:**
- Each session (identified by BlockId like `blk_123`) is labeled Normal or Anomaly
- Provided in `anomaly_label.csv`

**Template Mappings:**
- Each session's template sequence provided in `Event_traces.csv`
- Template patterns defined in `HDFS.log_templates.csv`

**Example:**
```csv
BlockId,Label,Features
blk_-3544583377289625738,Fail,"[E5,E22,E5,E11,E9,E20,E21]"
```
Session `blk_-3544583377289625738` is anomalous and contains templates E5, E22, E11, E9, E20, E21.

### Why HDFS?

1. **Established Benchmark**: Widely used in log analysis research
2. **Pre-parsed Templates**: No subjective template extraction
3. **Real-World Data**: Production Hadoop Distributed File System logs
4. **Diverse Patterns**: Mix of common operations and rare failures
5. **Publicly Available**: Reproducible results

**Dataset Access:**
- Download: [Zenodo (HDFS_v1.zip)](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)
- Repository: [Loghub GitHub](https://github.com/logpai/loghub)

---

## Evaluation Process

### Data Sampling

The full dataset (11M lines) takes hours to process. To enable rapid iteration, use random sampling:

1. Select random offset in log file
2. Read N consecutive lines (default: 100,000)
3. Extract sessions present in sample
4. Filter ground truth to sampled sessions only

### Template Attribution

**Identify anomaly templates in sample:**
```
For each session labeled "Anomaly":
  Add all templates from that session to anomaly_templates set
```

**Match flagged lines to templates:**
```
For each line flagged by the system:
  Match line to template using regex patterns
  If template is in anomaly_templates:
    Add to flagged_templates set
```

**Compute metrics:**
```
Template Recall = |flagged_templates| / |anomaly_templates|
```

### Statistical Validation

Run multiple evaluations to account for sampling variance. The seed controls random offsets for reproducibility. Output includes mean, standard deviation, and coefficient of variation (CV) for all metrics.

---

## Benchmark Scope

**What this measures:**
- Diversity of anomaly types detected
- Ability to find rare, unique patterns
- Coverage of template vocabulary

**What this doesn't measure:**
- Total number of anomalies detected
- Ability to find all instances of known errors
- Real-time detection latency
- Scalability to streaming logs

---

## Limitations

**Dataset:**
- Only validated on HDFS logs (29 templates over 11M lines). Generalization to other log types unknown.
- Requires pre-parsed templates for evaluation
- Ground truth labels entire sessions, not individual lines

**Evaluation:**
- Results may vary with different sample offsets
- No temporal metrics (detection latency, ordering)
- No severity weighting (all anomaly types treated equally)
- 100-occurrence threshold for "rare" templates is specific to HDFS dataset

---
