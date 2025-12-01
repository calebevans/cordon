from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from cordon.core.config import AnalysisConfig
from cordon.core.types import ScoredWindow, TextWindow

# Constants
_SCORING_PROGRESS_DESC = "Scoring embeddings   "


class DensityAnomalyScorer:
    """Calculate significance scores using k-NN cosine distance.

    This scorer uses the average distance to k nearest neighbors as a measure
    of how anomalous each window is. Higher distances
    indicate more anomalous content.

    For large datasets, automatically switches to memory-mapped storage to
    reduce RAM usage.
    """

    def _detect_device(self, config: AnalysisConfig) -> str:
        """Detect the best available device for scoring.

        Args:
            config: Analysis configuration with optional device setting

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if config.device is not None:
            return config.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _calculate_n_neighbors(self, config: AnalysisConfig, n_samples: int) -> int:
        """Calculate the number of neighbors to use for k-NN.

        Args:
            config: Analysis configuration with k_neighbors setting
            n_samples: Total number of samples in the dataset

        Returns:
            Number of neighbors to use (k+1 for self, capped at n_samples)
        """
        num_neighbors = config.k_neighbors
        return min(num_neighbors + 1, n_samples)

    def _score_windows_gpu(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
        device: str,
    ) -> list[ScoredWindow]:
        """Score windows using GPU acceleration (CUDA or MPS).

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration
            device: GPU device ('cuda' or 'mps')

        Returns:
            List of scored windows
        """
        # extract windows and embeddings
        windows = [window for window, _ in embedded_windows]
        embeddings_np = np.array([embedding for _, embedding in embedded_windows], dtype=np.float32)
        n_samples = len(embeddings_np)

        # convert to PyTorch tensor on GPU
        embeddings_tensor = torch.from_numpy(embeddings_np).to(device)

        # calculate number of neighbors
        n_neighbors = self._calculate_n_neighbors(config, n_samples)

        # process in batches to avoid GPU OOM
        query_batch_size = config.scoring_batch_size
        scored_windows = []

        for batch_start in tqdm(
            range(0, n_samples, query_batch_size),
            desc=_SCORING_PROGRESS_DESC,
            unit="batch",
            total=(n_samples + query_batch_size - 1) // query_batch_size,
        ):
            batch_end = min(batch_start + query_batch_size, n_samples)
            batch_embeddings = embeddings_tensor[batch_start:batch_end]

            # compute cosine similarity: similarity = batch @ all_embeddings.T
            # embeddings are already normalized, so this gives cosine similarity
            similarities = torch.mm(batch_embeddings, embeddings_tensor.T)

            # convert to distance: distance = 1 - similarity
            distances = 1.0 - similarities

            # find k+1 nearest neighbors (including self)
            neighbor_distances, _ = torch.topk(
                distances, k=n_neighbors, dim=1, largest=False, sorted=True
            )

            # move results back to CPU for processing
            neighbor_distances_cpu = neighbor_distances.cpu().numpy()

            # calculate scores for this batch
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                window = windows[global_idx]
                embedding = embeddings_np[global_idx]

                # skip first distance (self = 0) and take mean of remaining
                neighbor_dists = neighbor_distances_cpu[local_idx][1:]
                score = float(np.mean(neighbor_dists))

                scored_windows.append(ScoredWindow(window=window, score=score, embedding=embedding))

            # clear GPU cache after each batch for memory management
            if device == "cuda":
                torch.cuda.empty_cache()
            # MPS doesn't have empty_cache equivalent

        return scored_windows

    def _score_windows_cpu_pytorch(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows using PyTorch on CPU.

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration

        Returns:
            List of scored windows
        """
        # extract windows and embeddings
        windows = [window for window, _ in embedded_windows]
        embeddings_np = np.array([embedding for _, embedding in embedded_windows], dtype=np.float32)
        n_samples = len(embeddings_np)

        # convert to PyTorch tensor on CPU
        embeddings_tensor = torch.from_numpy(embeddings_np)

        # calculate number of neighbors
        n_neighbors = self._calculate_n_neighbors(config, n_samples)

        # process in batches
        query_batch_size = config.scoring_batch_size
        scored_windows = []

        for batch_start in tqdm(
            range(0, n_samples, query_batch_size),
            desc=_SCORING_PROGRESS_DESC,
            unit="batch",
            total=(n_samples + query_batch_size - 1) // query_batch_size,
        ):
            batch_end = min(batch_start + query_batch_size, n_samples)
            batch_embeddings = embeddings_tensor[batch_start:batch_end]

            # compute cosine similarity: similarity = batch @ all_embeddings.T
            similarities = torch.mm(batch_embeddings, embeddings_tensor.T)

            # convert to distance: distance = 1 - similarity
            distances = 1.0 - similarities

            # find k+1 nearest neighbors (including self)
            neighbor_distances, _ = torch.topk(
                distances, k=n_neighbors, dim=1, largest=False, sorted=True
            )

            # convert to numpy for processing
            neighbor_distances_np = neighbor_distances.numpy()

            # calculate scores for this batch
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                window = windows[global_idx]
                embedding = embeddings_np[global_idx]

                # skip first distance (self = 0) and take mean of remaining
                neighbor_dists = neighbor_distances_np[local_idx][1:]
                score = float(np.mean(neighbor_dists))

                scored_windows.append(ScoredWindow(window=window, score=score, embedding=embedding))

        return scored_windows

    def score_windows(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows based on k-NN density.

        This is the central routing function that selects the appropriate
        scoring implementation based on available hardware and configuration.

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration with k_neighbors setting

        Returns:
            List of scored windows with anomaly scores
        """
        if not embedded_windows:
            return []

        # single window
        if len(embedded_windows) == 1:
            window, embedding = embedded_windows[0]
            return [ScoredWindow(window=window, score=0.0, embedding=embedding)]

        # detect best available device
        device = self._detect_device(config)

        # route to appropriate PyTorch implementation
        if device in ("cuda", "mps"):
            # use GPU-accelerated implementation
            return self._score_windows_gpu(embedded_windows, config, device)
        else:
            # use PyTorch CPU implementation
            return self._score_windows_cpu_pytorch(embedded_windows, config)
