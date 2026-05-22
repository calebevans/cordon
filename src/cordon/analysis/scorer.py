"""Density anomaly scorer using k-NN cosine distance."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from cordon.core.config import AnalysisConfig
from cordon.core.types import ScoredWindow, TextWindow

logger = logging.getLogger(__name__)

_SCORING_PROGRESS_DESC = "Scoring embeddings   "

_CPU_DEFAULT_BATCH = 10_000
_MIN_BATCH = 1_000
_MAX_GPU_BATCH = 100_000
_MAX_MPS_BATCH = 50_000
_GPU_MEMORY_FRACTION = 0.1
_MPS_ASSUMED_MEMORY_GB = 1.0
_GPU_TARGET_MEMORY_GB = 1
_CPU_TARGET_MEMORY_GB = 2


class DensityAnomalyScorer:
    """Calculate significance scores using k-NN cosine distance.

    Uses the average distance to k nearest neighbors as a measure
    of how anomalous each window is. Higher distances indicate more
    anomalous content.
    """

    def _calculate_n_neighbors(self, config: AnalysisConfig, n_samples: int) -> int:
        """Calculate the number of neighbors to use for k-NN.

        Args:
            config: Analysis configuration with k_neighbors setting.
            n_samples: Total number of samples in the dataset.

        Returns:
            Number of neighbors to use (k+1 for self, capped at n_samples).
        """
        return min(config.k_neighbors + 1, n_samples)

    def _auto_detect_batch_size(self, n_samples: int, device: str) -> int:
        """Auto-detect optimal batch size based on available memory.

        Args:
            n_samples: Total number of samples in the dataset.
            device: Device string ('cuda', 'mps', or 'cpu').

        Returns:
            Optimal batch size for scoring.
        """
        if device == "cpu":
            return _CPU_DEFAULT_BATCH

        if device == "cuda":
            try:
                import torch

                props = torch.cuda.get_device_properties(0)
                total_memory_gb = props.total_memory / 1024**3
                target_memory_gb = total_memory_gb * _GPU_MEMORY_FRACTION
                batch_size = int((target_memory_gb * 1024**3) / (n_samples * 4))
                return max(_MIN_BATCH, min(batch_size, _MAX_GPU_BATCH))
            except Exception:
                logger.warning(
                    "Failed to auto-detect GPU batch size, using default",
                    exc_info=True,
                )
                return _CPU_DEFAULT_BATCH

        if device == "mps":
            batch_size = int((_MPS_ASSUMED_MEMORY_GB * 1024**3) / (n_samples * 4))
            return max(_MIN_BATCH, min(batch_size, _MAX_MPS_BATCH))

        return _CPU_DEFAULT_BATCH

    def _score_windows(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
        device: str,
    ) -> list[ScoredWindow]:
        """Score windows using k-NN density on the specified device.

        Args:
            embedded_windows: Sequence of (window, embedding) pairs.
            config: Analysis configuration.
            device: PyTorch device string ('cuda', 'mps', or 'cpu').

        Returns:
            List of scored windows with anomaly scores.
        """
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        windows = [window for window, _ in embedded_windows]
        embeddings_np = np.array([embedding for _, embedding in embedded_windows], dtype=np.float32)
        n_samples = len(embeddings_np)

        embeddings_tensor = torch.from_numpy(embeddings_np).to(device)
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

        n_neighbors = self._calculate_n_neighbors(config, n_samples)

        if config.scoring_batch_size is None:
            query_batch_size = self._auto_detect_batch_size(n_samples, device)
        else:
            query_batch_size = config.scoring_batch_size

        bytes_per_element = 4
        target_memory_gb = _GPU_TARGET_MEMORY_GB if device != "cpu" else _CPU_TARGET_MEMORY_GB
        chunk_size = int((target_memory_gb * 1024**3) / (query_batch_size * bytes_per_element))
        chunk_size = min(chunk_size, n_samples)
        chunk_size = max(chunk_size, 1)

        scored_windows: list[ScoredWindow] = []

        for batch_start in tqdm(
            range(0, n_samples, query_batch_size),
            desc=_SCORING_PROGRESS_DESC,
            unit="batch",
            total=(n_samples + query_batch_size - 1) // query_batch_size,
            disable=not config.show_progress,
        ):
            batch_end = min(batch_start + query_batch_size, n_samples)
            batch_embeddings = embeddings_tensor[batch_start:batch_end]
            batch_size_actual = batch_end - batch_start

            top_k_distances = torch.full(
                (batch_size_actual, n_neighbors),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )

            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_embeddings = embeddings_tensor[chunk_start:chunk_end]

                similarities = torch.mm(batch_embeddings, chunk_embeddings.T)
                # Clamp to handle floating point errors where cosine > 1.0
                chunk_distances = torch.clamp(1.0 - similarities, min=0.0, max=2.0)

                chunk_topk, _ = torch.topk(
                    chunk_distances,
                    k=min(n_neighbors, chunk_distances.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                combined = torch.cat([top_k_distances, chunk_topk], dim=1)
                top_k_distances, _ = torch.topk(
                    combined, k=n_neighbors, dim=1, largest=False, sorted=True
                )

            neighbor_distances_np = top_k_distances.cpu().numpy()
            batch_scores = np.maximum(neighbor_distances_np[:, 1:].mean(axis=1), 0.0)

            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                scored_windows.append(
                    ScoredWindow(
                        window=windows[global_idx],
                        score=float(batch_scores[local_idx]),
                    )
                )

        if device == "cuda":
            torch.cuda.empty_cache()

        return scored_windows

    def score_windows(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows based on k-NN density.

        Routes to the unified scoring implementation after detecting the
        best available compute device.

        Args:
            embedded_windows: Sequence of (window, embedding) pairs.
            config: Analysis configuration with k_neighbors setting.

        Returns:
            List of scored windows with anomaly scores.
        """
        if not embedded_windows:
            return []

        if len(embedded_windows) == 1:
            window, _ = embedded_windows[0]
            return [ScoredWindow(window=window, score=0.0)]

        from cordon.core.device import detect_device

        device = detect_device(config.device)
        return self._score_windows(embedded_windows, config, device)
