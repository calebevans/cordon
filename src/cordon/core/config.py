from dataclasses import dataclass
from pathlib import Path


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
        backend: Embedding backend to use ('sentence-transformers' or 'llama-cpp')
        model_path: Path to GGUF model file (required for llama-cpp backend)
        n_ctx: Context size for llama.cpp (default: 2048)
        n_threads: Thread count for llama.cpp (None=auto-detect)
        n_gpu_layers: Number of layers to offload to GPU in llama.cpp (0=CPU only)
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
    backend: str = "sentence-transformers"  # or "llama-cpp"
    model_path: str | None = None  # GGUF model file path
    n_ctx: int = 2048  # llama.cpp context size
    n_threads: int | None = None  # llama.cpp threads (None=auto)
    n_gpu_layers: int = 0  # llama.cpp GPU layer offloading

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

        # Backend validation
        if self.backend not in ("sentence-transformers", "llama-cpp"):
            raise ValueError(
                f"backend must be 'sentence-transformers' or 'llama-cpp', got '{self.backend}'"
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

        if self.n_gpu_layers < 0:
            raise ValueError("n_gpu_layers must be >= 0")

        if self.n_threads is not None and self.n_threads < 1:
            raise ValueError("n_threads must be >= 1 or None for auto-detect")
