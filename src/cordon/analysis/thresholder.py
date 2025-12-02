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
            config: Analysis configuration with anomaly_percentile or anomaly_range

        Returns:
            List of significant windows, sorted by score (descending)
        """
        # no scored windows
        if not scored_windows:
            return []

        # check if using range mode
        if config.anomaly_range_min is not None:
            # Range mode: exclude top X%, keep next Y%
            # Type narrowing: if min is not None, max is also not None (enforced in config)
            assert config.anomaly_range_max is not None

            scores = np.array([sw.score for sw in scored_windows])

            # Calculate percentile thresholds
            # e.g., min=0.05 (exclude top 5%) -> 95th percentile
            # e.g., max=0.15 (include up to 15%) -> 85th percentile
            upper_percentile = (1 - config.anomaly_range_min) * 100
            lower_percentile = (1 - config.anomaly_range_max) * 100

            upper_threshold = np.percentile(scores, upper_percentile)
            lower_threshold = np.percentile(scores, lower_percentile)

            # Select windows in the range: lower <= score < upper
            selected = [
                sw for sw in scored_windows if lower_threshold <= sw.score < upper_threshold
            ]

            # sort by score descending
            selected.sort(key=lambda window: window.score, reverse=True)

            return selected

        # Single percentile mode (original behavior)

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
