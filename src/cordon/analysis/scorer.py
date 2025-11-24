import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from cordon.core.config import AnalysisConfig
from cordon.core.types import ScoredWindow, TextWindow

# optional FAISS support
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class DensityAnomalyScorer:
    """Calculate significance scores using k-NN cosine distance.

    This scorer uses the average distance to k nearest neighbors as a measure
    of how anomalous each window is. Higher distances
    indicate more anomalous content.

    For large datasets, automatically switches to memory-mapped storage to
    reduce RAM usage.
    """

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

    def score_windows(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows based on k-NN density.

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

        n_windows = len(embedded_windows)

        # choose strategy based on dataset size
        use_faiss = (
            HAS_FAISS
            and config.use_faiss_threshold is not None
            and n_windows >= config.use_faiss_threshold
        )
        use_mmap = (
            config.use_mmap_threshold is not None
            and n_windows >= config.use_mmap_threshold
            and not use_faiss  # FAISS takes precedence
        )

        if use_faiss:
            return self._score_windows_faiss(embedded_windows, config)
        elif use_mmap:
            return self._score_windows_mmap(embedded_windows, config)
        else:
            return self._score_windows_inmemory(embedded_windows, config)

    def _score_windows_inmemory(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows using in-memory arrays (fast, but uses more RAM).

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration

        Returns:
            List of scored windows
        """
        # extract embeddings into matrix
        windows = [window for window, _ in embedded_windows]
        embeddings = np.array([embedding for _, embedding in embedded_windows])

        # build k-NN index
        n_samples = len(embeddings)
        n_neighbors = self._calculate_n_neighbors(config, n_samples)

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(embeddings)

        # query all points
        distances, _ = knn.kneighbors(embeddings)

        # calculate scores (average distance to k nearest neighbors, excluding self)
        scored_windows = []
        for window_idx, (window, embedding) in enumerate(zip(windows, embeddings, strict=False)):
            # skip first distance (self = 0) and take mean of remaining
            neighbor_distances = distances[window_idx][1:]
            score = float(np.mean(neighbor_distances))

            scored_windows.append(ScoredWindow(window=window, score=score, embedding=embedding))

        return scored_windows

    def _score_windows_mmap(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows using memory-mapped storage (lower RAM, slightly slower).

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration

        Returns:
            List of scored windows
        """
        windows = [window for window, _ in embedded_windows]
        n_windows = len(windows)

        # embedding dimension from first embedding
        first_embedding = embedded_windows[0][1]
        embedding_dim = len(first_embedding)

        # create temporary memory-mapped file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            # memory-mapped array for embeddings
            embeddings_mmap = np.memmap(
                temp_path,
                dtype="float32",
                mode="w+",
                shape=(n_windows, embedding_dim),
            )

            # copy embeddings to mmap and flush to disk
            for window_idx, (_, embedding) in enumerate(embedded_windows):
                embeddings_mmap[window_idx] = embedding

            embeddings_mmap.flush()

            # build k-NN index
            n_neighbors = self._calculate_n_neighbors(config, n_windows)

            knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            knn.fit(embeddings_mmap)

            # query all points and calculate scores
            distances, _ = knn.kneighbors(embeddings_mmap)
            scored_windows = []
            for window_idx, window in enumerate(windows):
                neighbor_distances = distances[window_idx][1:]
                score = float(np.mean(neighbor_distances))

                scored_windows.append(
                    ScoredWindow(
                        window=window,
                        score=score,
                        embedding=embeddings_mmap[window_idx].copy(),
                    )
                )

            return scored_windows

        finally:
            # clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def _score_windows_faiss(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: AnalysisConfig,
    ) -> list[ScoredWindow]:
        """Score windows using FAISS for fast approximate k-NN (lowest RAM, fastest).

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration

        Returns:
            List of scored windows
        """
        if not HAS_FAISS:
            warnings.warn(
                "FAISS not available, falling back to memory-mapped approach. "
                "Install faiss-cpu or faiss-gpu for better performance on large logs.",
                UserWarning,
                stacklevel=2,
            )
            return self._score_windows_mmap(embedded_windows, config)

        windows = [window for window, _ in embedded_windows]
        embeddings = np.array([embedding for _, embedding in embedded_windows], dtype=np.float32)

        n_windows = len(embeddings)
        embedding_dim = embeddings.shape[1]
        n_neighbors = self._calculate_n_neighbors(config, n_windows)

        # normalize embeddings so inner product = cosine similarity
        faiss.normalize_L2(embeddings)

        # create FAISS index
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)

        # query k-nearest neighbors
        distances, _ = index.search(embeddings, n_neighbors)

        # convert inner product (cosine similarity) to cosine distance
        # after normalization, inner product equals cosine similarity
        distances = 1.0 - distances

        # calculate scores
        scored_windows = []
        for window_idx, window in enumerate(windows):
            # skip first distance (self) and take mean of remaining
            neighbor_distances = distances[window_idx][1:]
            score = float(np.mean(neighbor_distances))

            # ensure non-negative scores (handle numerical precision issues)
            score = max(0.0, score)

            scored_windows.append(
                ScoredWindow(window=window, score=score, embedding=embeddings[window_idx])
            )

        return scored_windows
