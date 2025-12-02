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

        Raises:
            RuntimeError: If CUDA device is incompatible
        """
        if config.device is not None:
            device = config.device
            # validate CUDA compatibility if CUDA requested
            if device == "cuda" and torch.cuda.is_available():
                self._check_cuda_compatibility()
            return device

        # auto-detect device priority: cuda > mps > cpu
        if torch.cuda.is_available():
            self._check_cuda_compatibility()
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _check_cuda_compatibility(self) -> None:
        """Check if CUDA GPU is compatible with current PyTorch build.

        Raises:
            RuntimeError: If GPU compute capability is too old for PyTorch
        """
        if not torch.cuda.is_available():
            return

        # get compute capability
        device_props = torch.cuda.get_device_properties(0)

        # PyTorch 2.0+ requires compute capability >= 6.0 (Pascal or newer)
        if device_props.major < 6:
            gpu_name = device_props.name
            compute_capability = f"{device_props.major}.{device_props.minor}"
            raise RuntimeError(
                f"GPU {gpu_name} (compute capability {compute_capability}) is not supported by "
                f"PyTorch 2.0+. Requires compute capability >= 6.0 (Pascal architecture or newer)."
            )

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

    def _auto_detect_batch_size(self, n_samples: int, device: str) -> int:
        """Auto-detect optimal batch size based on available memory.

        Args:
            n_samples: Total number of samples in the dataset
            device: Device string ('cuda', 'mps', or 'cpu')

        Returns:
            Optimal batch size for scoring
        """
        if device == "cpu":
            # CPU: use conservative default
            return 10000

        if device == "cuda":
            try:
                # Get available GPU memory
                props = torch.cuda.get_device_properties(0)
                total_memory_gb = props.total_memory / 1024**3

                # Target: use ~10% of GPU memory for distance matrix
                # distance_matrix = batch_size × n_samples × 4 bytes (float32)
                target_memory_gb = total_memory_gb * 0.1
                batch_size = int((target_memory_gb * 1024**3) / (n_samples * 4))

                # reasonable bounds: 1k to 100k
                return max(1000, min(batch_size, 100000))
            except Exception:
                return 10000

        if device == "mps":
            # MPS: conservative estimate (Apple doesn't expose memory info easily)
            # assume ~10GB available, use similar calculation
            target_memory_gb = 1.0  # 10% of assumed 10GB
            batch_size = int((target_memory_gb * 1024**3) / (n_samples * 4))
            return max(1000, min(batch_size, 50000))

        # fallback
        return 10000

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

        # determine batch size (auto-detect if not specified)
        if config.scoring_batch_size is None:
            query_batch_size = self._auto_detect_batch_size(n_samples, device)
        else:
            query_batch_size = config.scoring_batch_size

        scored_windows = []

        # compute chunk size for similarity calculation to avoid OOM
        # aim for ~1GB of similarity matrix per chunk
        bytes_per_element = 4  # float32
        target_memory_gb = 1
        chunk_size = int((target_memory_gb * 1024**3) / (query_batch_size * bytes_per_element))
        chunk_size = min(chunk_size, n_samples)  # don't exceed total samples

        for batch_start in tqdm(
            range(0, n_samples, query_batch_size),
            desc=_SCORING_PROGRESS_DESC,
            unit="batch",
            total=(n_samples + query_batch_size - 1) // query_batch_size,
        ):
            batch_end = min(batch_start + query_batch_size, n_samples)
            batch_embeddings = embeddings_tensor[batch_start:batch_end]
            batch_size_actual = batch_end - batch_start

            # track top-k distances across chunks (more memory efficient)
            # initialize with large values
            top_k_distances = torch.full(
                (batch_size_actual, n_neighbors), float("inf"), dtype=torch.float32, device=device
            )

            # compute similarities in chunks and maintain top-k
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_embeddings = embeddings_tensor[chunk_start:chunk_end]

                # compute cosine similarity for this chunk
                similarities = torch.mm(batch_embeddings, chunk_embeddings.T)

                # convert to distance (clamp to handle floating point errors)
                # cosine similarity can be slightly > 1.0 due to floating point precision
                chunk_distances = torch.clamp(1.0 - similarities, min=0.0, max=2.0)

                # combine with existing top-k
                combined_distances = torch.cat([top_k_distances, chunk_distances], dim=1)

                # find new top-k from combined
                top_k_distances, _ = torch.topk(
                    combined_distances, k=n_neighbors, dim=1, largest=False, sorted=True
                )

            # move results back to CPU for processing
            neighbor_distances_cpu = top_k_distances.cpu().numpy()

            # calculate scores for this batch
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                window = windows[global_idx]
                embedding = embeddings_np[global_idx]

                # skip first distance (self = 0) and take mean of remaining
                neighbor_dists = neighbor_distances_cpu[local_idx][1:]
                score = float(np.mean(neighbor_dists))

                # ensure score is non-negative (handle any remaining floating point errors)
                score = max(0.0, score)

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

        # determine batch size (auto-detect if not specified)
        if config.scoring_batch_size is None:
            query_batch_size = self._auto_detect_batch_size(n_samples, "cpu")
        else:
            query_batch_size = config.scoring_batch_size

        scored_windows = []

        # compute chunk size for similarity calculation to avoid OOM on CPU
        # aim for ~2GB of similarity matrix per chunk on CPU (more than GPU)
        bytes_per_element = 4  # float32
        target_memory_gb = 2
        chunk_size = int((target_memory_gb * 1024**3) / (query_batch_size * bytes_per_element))
        chunk_size = min(chunk_size, n_samples)  # don't exceed total samples

        for batch_start in tqdm(
            range(0, n_samples, query_batch_size),
            desc=_SCORING_PROGRESS_DESC,
            unit="batch",
            total=(n_samples + query_batch_size - 1) // query_batch_size,
        ):
            batch_end = min(batch_start + query_batch_size, n_samples)
            batch_embeddings = embeddings_tensor[batch_start:batch_end]
            batch_size_actual = batch_end - batch_start

            # track top-k distances across chunks (more memory efficient)
            # initialize with large values
            top_k_distances = torch.full(
                (batch_size_actual, n_neighbors), float("inf"), dtype=torch.float32
            )

            # compute similarities in chunks and maintain top-k
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_embeddings = embeddings_tensor[chunk_start:chunk_end]

                # compute cosine similarity for this chunk
                similarities = torch.mm(batch_embeddings, chunk_embeddings.T)

                # convert to distance (clamp to handle floating point errors)
                # cosine similarity can be slightly > 1.0 due to floating point precision
                chunk_distances = torch.clamp(1.0 - similarities, min=0.0, max=2.0)

                # combine with existing top-k
                combined_distances = torch.cat([top_k_distances, chunk_distances], dim=1)

                # find new top-k from combined
                top_k_distances, _ = torch.topk(
                    combined_distances, k=n_neighbors, dim=1, largest=False, sorted=True
                )

            # convert to numpy for processing
            neighbor_distances_np = top_k_distances.numpy()

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
