from collections.abc import Sequence

import numpy as np

from cordon.core.config import AnalysisConfig
from cordon.core.types import ScoredWindow


class Thresholder:
    """Select top windows based on anomaly percentile.

    Determines which windows are significant based on the distribution
    of scores in the current dataset.
    """

    def select_significant(
        self, scored_windows: Sequence[ScoredWindow], config: AnalysisConfig
    ) -> list[ScoredWindow]:
        """Select significant windows based on threshold.

        Args:
            scored_windows: Sequence of scored windows
            config: Analysis configuration with anomaly_percentile

        Returns:
            List of significant windows, sorted by score (descending)
        """
        # no scored windows
        if not scored_windows:
            return []

        # all windows, sorted by score descending
        if config.anomaly_percentile == 1.0:
            return sorted(scored_windows, key=lambda window: window.score, reverse=True)

        # no windows requested
        if config.anomaly_percentile == 0.0:
            return []

        # calculate percentile threshold
        scores = np.array([sw.score for sw in scored_windows])
        percentile = (1 - config.anomaly_percentile) * 100
        threshold = np.percentile(scores, percentile)

        # filter windows at or above threshold
        selected = [sw for sw in scored_windows if sw.score >= threshold]

        # sort by score descending (highest anomalies first)
        selected.sort(key=lambda window: window.score, reverse=True)

        return selected
