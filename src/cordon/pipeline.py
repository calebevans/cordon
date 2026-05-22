import time
from pathlib import Path

import numpy as np

from cordon.analysis.scorer import DensityAnomalyScorer
from cordon.analysis.thresholder import Thresholder as ThresholderImpl
from cordon.core.config import AnalysisConfig
from cordon.core.types import (
    AnalysisResult,
    Formatter,
    Merger,
    Reader,
    ScoredWindow,
    Scorer,
    Segmenter,
    Thresholder,
)
from cordon.embedding import create_embedder
from cordon.ingestion.reader import LogFileReader
from cordon.postprocess.formatter import XmlFormatter
from cordon.postprocess.merger import IntervalMerger
from cordon.segmentation.windower import SlidingWindowSegmenter


class SemanticLogAnalyzer:
    """High-level API for semantic log analysis.

    This class orchestrates the complete analysis pipeline, from reading
    log files through to generating formatted output with significant
    anomalies highlighted.
    """

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        *,
        reader: Reader | None = None,
        segmenter: Segmenter | None = None,
        scorer: Scorer | None = None,
        thresholder: Thresholder | None = None,
        merger: Merger | None = None,
        formatter: Formatter | None = None,
    ) -> None:
        """Initialize the analyzer with configuration and optional custom components.

        Args:
            config: Analysis configuration (uses defaults if None).
            reader: Custom reader implementation.
            segmenter: Custom segmenter implementation.
            scorer: Custom scorer implementation.
            thresholder: Custom thresholder implementation.
            merger: Custom merger implementation.
            formatter: Custom formatter implementation.
        """
        self.config = config if config is not None else AnalysisConfig()
        self._embedder = create_embedder(self.config)
        self._reader = reader if reader is not None else LogFileReader()
        self._segmenter = segmenter if segmenter is not None else SlidingWindowSegmenter()
        self._scorer = scorer if scorer is not None else DensityAnomalyScorer()
        self._thresholder = thresholder if thresholder is not None else ThresholderImpl()
        self._merger = merger if merger is not None else IntervalMerger()
        self._formatter = formatter if formatter is not None else XmlFormatter()

    def analyze_file(self, file_path: Path) -> str:
        """Analyze a log file and return formatted output.

        Args:
            file_path: Path to the log file to analyze.

        Returns:
            Formatted string with XML-tagged significant blocks.
        """
        result = self.analyze_file_detailed(file_path)
        return result.output

    def analyze_file_detailed(self, file_path: Path) -> AnalysisResult:
        """Analyze a log file and return detailed results.

        Args:
            file_path: Path to the log file to analyze.

        Returns:
            Complete analysis result with metadata.
        """
        start_time = time.time()

        # stage 1: ingestion
        lines_list = list(self._reader.read_lines(file_path))
        total_lines = len(lines_list)

        # stage 2: segmentation
        windows = self._segmenter.segment(iter(lines_list), self.config)

        # stage 3: vectorization
        embedded = list(self._embedder.embed_windows(windows))
        total_windows = len(embedded)

        # stage 4: scoring
        scored = self._scorer.score_windows(embedded, self.config)
        del embedded

        # stage 5: thresholding
        significant = self._thresholder.select_significant(scored, self.config)
        significant_windows = len(significant)

        # stage 6: merging
        merged = self._merger.merge_windows(significant)
        merged_blocks = len(merged)
        del significant

        # stage 7: formatting
        output = self._formatter.format_blocks(merged, lines_list)

        # calculate statistics
        processing_time = time.time() - start_time
        score_distribution = self._calculate_score_distribution(scored)
        del scored

        return AnalysisResult(
            output=output,
            total_lines=total_lines,
            total_windows=total_windows,
            significant_windows=significant_windows,
            merged_blocks=merged_blocks,
            score_distribution=score_distribution,
            processing_time=processing_time,
        )

    def _calculate_score_distribution(self, scored_windows: list[ScoredWindow]) -> dict[str, float]:
        """Calculate statistical distribution of scores.

        Args:
            scored_windows: List of scored windows.

        Returns:
            Dictionary with statistical measures.
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
