import time
from pathlib import Path

import numpy as np

from cordon.analysis.scorer import DensityAnomalyScorer
from cordon.analysis.thresholder import Thresholder
from cordon.core.config import AnalysisConfig
from cordon.core.types import AnalysisResult, ScoredWindow
from cordon.embedding import create_vectorizer
from cordon.ingestion.reader import LogFileReader
from cordon.postprocess.formatter import OutputFormatter
from cordon.postprocess.merger import IntervalMerger
from cordon.segmentation.windower import SlidingWindowSegmenter


class SemanticLogAnalyzer:
    """High-level API for semantic log analysis.

    This class orchestrates the complete analysis pipeline, from reading
    log files through to generating formatted output with significant
    anomalies highlighted.
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize the analyzer with configuration.

        Args:
            config: Analysis configuration (uses defaults if None)
        """
        self.config = config if config is not None else AnalysisConfig()

    def analyze_file(self, file_path: Path) -> str:
        """Analyze a log file and return formatted output.

        Args:
            file_path: Path to the log file to analyze

        Returns:
            Formatted string with XML-tagged significant blocks
        """
        result = self.analyze_file_detailed(file_path)
        return result.output

    def analyze_file_detailed(self, file_path: Path) -> AnalysisResult:
        """Analyze a log file and return detailed results.

        Args:
            file_path: Path to the log file to analyze

        Returns:
            Complete analysis result with metadata
        """
        start_time = time.time()

        # stage 1: ingestion
        reader = LogFileReader()
        lines = reader.read_lines(file_path)

        # stage 2: segmentation
        segmenter = SlidingWindowSegmenter()
        windows = segmenter.segment(lines, self.config)

        # stage 3: vectorization (using factory to select backend)
        vectorizer = create_vectorizer(self.config)
        embedded = list(vectorizer.embed_windows(windows))
        total_windows = len(embedded)

        # stage 4: scoring
        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, self.config)

        # stage 5: thresholding
        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, self.config)
        significant_windows = len(significant)

        # stage 6: merging
        merger = IntervalMerger()
        merged = merger.merge_windows(significant)
        merged_blocks = len(merged)

        # stage 7: formatting
        formatter = OutputFormatter()
        output = formatter.format_blocks(merged, file_path)

        # calculate statistics
        processing_time = time.time() - start_time
        score_distribution = self._calculate_score_distribution(scored)

        return AnalysisResult(
            output=output,
            total_windows=total_windows,
            significant_windows=significant_windows,
            merged_blocks=merged_blocks,
            score_distribution=score_distribution,
            processing_time=processing_time,
        )

    def _calculate_score_distribution(self, scored_windows: list[ScoredWindow]) -> dict[str, float]:
        """Calculate statistical distribution of scores.

        Args:
            scored_windows: List of scored windows

        Returns:
            Dictionary with statistical measures
        """
        if not scored_windows:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p90": 0.0,
            }

        scores = np.array([sw.score for sw in scored_windows])

        return {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
        }
