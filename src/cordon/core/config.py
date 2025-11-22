from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Global configuration for the analysis pipeline.

    Attributes:
        window_size: Number of lines per window
        stride: Step size for sliding window (number of lines to skip)
        k_neighbors: Number of neighbors for k-NN density calculation
        anomaly_percentile: Percentile of windows to retain (e.g., 0.1 = top 10%)
        model_name: Name of sentence-transformers model to use
        batch_size: Batch size for embedding operations
        device: Device for model inference ('cuda', 'mps', 'cpu', or None for auto)
        use_mmap_threshold: Auto-enable memory mapping above this many windows (None=auto)
        use_faiss_threshold: Auto-enable FAISS above this many windows (None=auto)
    """

    window_size: int = 5
    stride: int = 2
    k_neighbors: int = 5
    anomaly_percentile: float = 0.1
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str | None = None
    use_mmap_threshold: int | None = 50000  # switch to mmap at 50k windows
    use_faiss_threshold: int | None = None  # FAISS disabled by default

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if not 0.0 <= self.anomaly_percentile <= 1.0:
            raise ValueError("anomaly_percentile must be between 0.0 and 1.0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.device is not None and self.device not in ("cuda", "mps", "cpu"):
            raise ValueError("device must be 'cuda', 'mps', 'cpu', or None")
