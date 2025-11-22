# Real-World Test Examples

This directory contains comprehensive test results of Cordon running against diverse real-world log datasets from the [LogHub](https://github.com/logpai/loghub) repository - a large collection of system log datasets for AI-driven log analytics research.

## Test Overview

All tests were run using Cordon's semantic anomaly detection to identify unusual patterns in 2,000-line log samples across different system types. The tool successfully reduced each 2,000-line log to an average of **306 lines across 18.5 blocks** (84.7% reduction), demonstrating its effectiveness for log summarization and LLM pre-processing.

## Test Results Summary

| Test Name | Log Type | Source Dataset | Cordon Output |
|-----------|----------|----------------|---------------|
| **HDFS** | Distributed Filesystem | [HDFS_2k.log](https://github.com/logpai/loghub/blob/master/HDFS/HDFS_2k.log) | [View Output](./output/hdfs-output.txt) |
| **Apache** | Web Server Errors | [Apache_2k.log](https://github.com/logpai/loghub/blob/master/Apache/Apache_2k.log) | [View Output](./output/apache-output.txt) |
| **Linux** | Operating System | [Linux_2k.log](https://github.com/logpai/loghub/blob/master/Linux/Linux_2k.log) | [View Output](./output/linux-output.txt) |
| **OpenSSH** | SSH Server | [OpenSSH_2k.log](https://github.com/logpai/loghub/blob/master/OpenSSH/OpenSSH_2k.log) | [View Output](./output/openssh-output.txt) |
| **Hadoop** | MapReduce Jobs | [Hadoop_2k.log](https://github.com/logpai/loghub/blob/master/Hadoop/Hadoop_2k.log) | [View Output](./output/hadoop-output.txt) |
| **Windows** | Windows Events | [Windows_2k.log](https://github.com/logpai/loghub/blob/master/Windows/Windows_2k.log) | [View Output](./output/windows-output.txt) |
| **Zookeeper** | Distributed Coordination | [Zookeeper_2k.log](https://github.com/logpai/loghub/blob/master/Zookeeper/Zookeeper_2k.log) | [View Output](./output/zookeeper-output.txt) |
| **Android** | Mobile Framework | [Android_2k.log](https://github.com/logpai/loghub/blob/master/Android/Android_2k.log) | [View Output](./output/android-output.txt) |
| **Spark** | Big Data Processing | [Spark_2k.log](https://github.com/logpai/loghub/blob/master/Spark/Spark_2k.log) | [View Output](./output/spark-output.txt) |
| **Proxifier** | Network Proxy | [Proxifier_2k.log](https://github.com/logpai/loghub/blob/master/Proxifier/Proxifier_2k.log) | [View Output](./output/proxifier-output.txt)* |

\* *Proxifier test used custom parameters: `--window-size 20 --k-neighbors 10 --anomaly-percentile 0.05`*

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Average Processing Time** | ~3.05 seconds per 2,000-line file |
| **Processing Range** | 2.94s - 3.33s |
| **Average Reduction Ratio** | 2,000 lines → 306 lines in 18.5 blocks (84.7% reduction) |
| **Model Used** | all-MiniLM-L6-v2 (384-dimensional embeddings) |

## Interpreting the Output

Each output file contains:

1. **Analysis Statistics**: Summary metrics about the analysis
   - Total windows created
   - Significant windows identified
   - Merged blocks (final output count)
   - Processing time
   
2. **Score Distribution**: Statistical overview of anomaly scores
   - Min, Mean, Median, P90, Max scores
   - Higher scores indicate more semantically unique content

3. **Significant Blocks**: The actual anomalous log sections
   - Line ranges (e.g., `lines="1-25"`)
   - Anomaly score for each block
   - Full log content of the anomalous section

## Example: Understanding a Result

From the OpenSSH output:

```xml
<block lines="1-20" score="0.0807">
Dec 10 06:55:46 LabSZ sshd[24200]: reverse mapping checking getaddrinfo for ns.marryaldkfaczcz.com [173.234.31.186] failed - POSSIBLE BREAK-IN ATTEMPT!
Dec 10 06:55:46 LabSZ sshd[24200]: Invalid user webmaster from 173.234.31.186
...
```

**Interpretation**:
- **Score 0.0807**: Moderately high, indicating this pattern is unusual
- **Content**: Reverse DNS failure + invalid user attempts = likely attack attempt
- **Lines 1-20**: Check these specific lines in the original log for context

## Running These Tests Yourself

To reproduce any of these tests:

```bash
# Install cordon
pip install -e .

# Clone the LogHub repository
git clone https://github.com/logpai/loghub.git

# Run a test (example: HDFS)
cordon --detailed loghub/HDFS/HDFS_2k.log

# With custom parameters (example: Proxifier)
cordon --window-size 20 --k-neighbors 10 --anomaly-percentile 0.05 --detailed loghub/Proxifier/Proxifier_2k.log
```

## Configuration Used

### Default Tests (HDFS, Apache, Linux, OpenSSH, Hadoop, Windows, Zookeeper, Android, Spark)
```
window_size:          10 lines
stride:               5 lines
k_neighbors:          5
anomaly_percentile:   0.1 (top 10%)
model:                all-MiniLM-L6-v2
```

### Proxifier Test (Custom Configuration)
```
window_size:          20 lines
stride:               5 lines (default)
k_neighbors:          10
anomaly_percentile:   0.05 (top 5%)
model:                all-MiniLM-L6-v2
```
---

**Last Updated**: November 22, 2025  
**Test Version**: Cordon v0.1.0  
**Python Version**: 3.12.10

