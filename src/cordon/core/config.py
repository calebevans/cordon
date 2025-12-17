from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisConfig:
    """Global configuration for the analysis pipeline."""

    window_size: int = 4
    k_neighbors: int = 5
    anomaly_percentile: float = 0.1
    anomaly_range_min: float | None = (
        None  # Lower bound for range mode (e.g., 0.05 = exclude top 5%)
    )
    anomaly_range_max: float | None = (
        None  # Upper bound for range mode (e.g., 0.15 = include up to 15%)
    )
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str | None = None
    scoring_batch_size: int | None = None  # batch size for k-NN scoring (None=auto-detect)
    backend: str = "sentence-transformers"  # or "llama-cpp" or "remote"
    model_path: str | None = None  # GGUF model file path
    n_ctx: int = 2048  # llama.cpp context size
    n_threads: int | None = None  # llama.cpp threads (None=auto)
    n_gpu_layers: int = 0  # llama.cpp GPU layer offloading
    api_key: str | None = None  # API key for remote embeddings
    endpoint: str | None = None  # Custom API endpoint URL
    request_timeout: float = 60.0  # Request timeout in seconds

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if not 0.0 <= self.anomaly_percentile <= 1.0:
            raise ValueError("anomaly_percentile must be between 0.0 and 1.0")

        # Validate anomaly range parameters
        if (self.anomaly_range_min is None) != (self.anomaly_range_max is None):
            raise ValueError(
                "anomaly_range_min and anomaly_range_max must both be set or both be None"
            )

        if self.anomaly_range_min is not None:
            # Type narrowing: if min is not None, max is also not None (checked above)
            assert self.anomaly_range_max is not None

            if not 0.0 <= self.anomaly_range_min <= 1.0:
                raise ValueError("anomaly_range_min must be between 0.0 and 1.0")
            if not 0.0 <= self.anomaly_range_max <= 1.0:
                raise ValueError("anomaly_range_max must be between 0.0 and 1.0")
            if self.anomaly_range_min >= self.anomaly_range_max:
                raise ValueError("anomaly_range_min must be less than anomaly_range_max")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.scoring_batch_size is not None and self.scoring_batch_size < 1:
            raise ValueError("scoring_batch_size must be >= 1 or None for auto-detect")
        if self.device is not None and self.device not in ("cuda", "mps", "cpu"):
            raise ValueError("device must be 'cuda', 'mps', 'cpu', or None")

        # Backend validation
        if self.backend not in ("sentence-transformers", "llama-cpp", "remote"):
            raise ValueError(
                f"backend must be 'sentence-transformers', 'llama-cpp', or 'remote', got '{self.backend}'"
            )

        # llama-cpp specific validation
        if self.backend == "llama-cpp" and self.model_path is not None:
            # If model_path is provided, validate it exists and has correct extension
            # If None, LlamaCppVectorizer will auto-download default model
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise ValueError(f"GGUF model file not found: {self.model_path}")

            if model_file.suffix != ".gguf":
                raise ValueError(f"model_path must be a .gguf file, got: {model_file.suffix}")

        # llama.cpp parameter validation
        if self.n_ctx < 1:
            raise ValueError("n_ctx must be >= 1")

        if self.n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be >= -1 (-1 for all layers, 0 for CPU-only)")

        if self.n_threads is not None and self.n_threads < 1:
            raise ValueError("n_threads must be >= 1 or None for auto-detect")

        # remote backend validation
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
