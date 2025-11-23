"""Unit tests for llama.cpp vectorizer backend."""

import numpy as np
import pytest

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class TestLlamaCppVectorizerConfiguration:
    """Tests for LlamaCppVectorizer configuration and initialization."""

    def test_missing_model_path_auto_downloads(self, monkeypatch, tmp_path) -> None:
        """Test that missing model_path triggers auto-download."""
        pytest.importorskip("llama_cpp")

        def mock_hf_hub_download(repo_id, filename):
            model_file = tmp_path / filename
            model_file.write_text("fake model")
            return str(model_file)

        import sys
        from unittest.mock import MagicMock

        mock_hub = MagicMock()
        mock_hub.hf_hub_download = mock_hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hub)

        config = AnalysisConfig(backend="llama-cpp", model_path=None)

        from cordon.embedding.llama_cpp import LlamaCppVectorizer

        with pytest.raises((RuntimeError, ValueError)):
            LlamaCppVectorizer(config)

    def test_nonexistent_model_path_raises_error(self) -> None:
        """Test that nonexistent model file raises ValueError."""
        with pytest.raises(ValueError, match="GGUF model file not found"):
            AnalysisConfig(backend="llama-cpp", model_path="/nonexistent/model.gguf")

    def test_import_error_without_llama_cpp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Test that ImportError is raised when llama-cpp-python is not installed."""
        import sys

        monkeypatch.setitem(sys.modules, "llama_cpp", None)

        model_file = tmp_path / "model.gguf"
        model_file.touch()

        config = AnalysisConfig(backend="llama-cpp", model_path=str(model_file))

        assert config.backend == "llama-cpp"
        assert config.model_path == str(model_file)


class TestLlamaCppVectorizerFactory:
    """Tests for factory function with llama.cpp backend."""

    def test_factory_creates_llama_vectorizer(self, tmp_path) -> None:
        """Test that factory function creates LlamaCppVectorizer."""
        pytest.importorskip("llama_cpp")

        from cordon.embedding import create_vectorizer

        model_file = tmp_path / "model.gguf"
        model_file.touch()

        config = AnalysisConfig(
            backend="llama-cpp",
            model_path=str(model_file),
        )

        with pytest.raises((ValueError, RuntimeError)):
            create_vectorizer(config)

    def test_factory_with_invalid_backend_raises_error(self) -> None:
        """Test that factory rejects invalid backend names."""
        with pytest.raises(ValueError, match="backend must be"):
            AnalysisConfig(backend="invalid-backend")


class TestLlamaCppVectorizerEmbedding:
    """Tests for LlamaCppVectorizer embedding functionality."""

    @pytest.fixture
    def model_path(self) -> str:
        """Get test model path from environment or skip."""
        import os

        model_path = os.environ.get("CORDON_TEST_GGUF_MODEL")
        if not model_path:
            pytest.skip("CORDON_TEST_GGUF_MODEL environment variable not set")
        return model_path

    @pytest.fixture
    def vectorizer(self, model_path: str):
        """Create a LlamaCppVectorizer instance for testing."""
        pytest.importorskip("llama_cpp")

        from cordon.embedding.llama_cpp import LlamaCppVectorizer

        config = AnalysisConfig(
            backend="llama-cpp",
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=0,
        )
        return LlamaCppVectorizer(config)

    def test_embed_single_window(self, vectorizer) -> None:
        """Test embedding a single text window."""
        window = TextWindow(
            content="Error: Connection timeout",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        results = list(vectorizer.embed_windows([window]))

        assert len(results) == 1
        result_window, embedding = results[0]
        assert result_window == window
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_embed_multiple_windows(self, vectorizer) -> None:
        """Test embedding multiple text windows."""
        windows = [
            TextWindow(
                content="Error: Connection timeout",
                start_line=1,
                end_line=1,
                window_id=0,
            ),
            TextWindow(
                content="Warning: Slow query detected",
                start_line=2,
                end_line=2,
                window_id=1,
            ),
            TextWindow(
                content="Info: Application started",
                start_line=3,
                end_line=3,
                window_id=2,
            ),
        ]

        results = list(vectorizer.embed_windows(windows))

        assert len(results) == 3
        for i, (result_window, embedding) in enumerate(results):
            assert result_window == windows[i]
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32

    def test_embedding_normalization(self, vectorizer) -> None:
        """Test that embeddings are L2 normalized."""
        window = TextWindow(
            content="Test content for normalization",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        results = list(vectorizer.embed_windows([window]))
        _, embedding = results[0]

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_embedding_consistency(self, vectorizer) -> None:
        """Test that same input produces same embedding."""
        window = TextWindow(
            content="Consistent content test",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        results1 = list(vectorizer.embed_windows([window]))
        results2 = list(vectorizer.embed_windows([window]))

        _, embedding1 = results1[0]
        _, embedding2 = results2[0]

        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)

    def test_empty_windows_list(self, vectorizer) -> None:
        """Test embedding empty list of windows."""
        results = list(vectorizer.embed_windows([]))
        assert len(results) == 0

    def test_semantic_similarity(self, vectorizer) -> None:
        """Test that semantically similar texts have similar embeddings."""
        window1 = TextWindow(
            content="Error: Database connection failed",
            start_line=1,
            end_line=1,
            window_id=0,
        )
        window2 = TextWindow(
            content="Error: Database connection timeout",
            start_line=2,
            end_line=2,
            window_id=1,
        )
        window3 = TextWindow(
            content="Info: User logged in successfully",
            start_line=3,
            end_line=3,
            window_id=2,
        )

        results = list(vectorizer.embed_windows([window1, window2, window3]))
        _, emb1 = results[0]
        _, emb2 = results[1]
        _, emb3 = results[2]

        sim_1_2 = np.dot(emb1, emb2)
        sim_1_3 = np.dot(emb1, emb3)

        assert sim_1_2 > sim_1_3


class TestLlamaCppVectorizerIntegration:
    """Integration tests with the full analysis pipeline."""

    def test_config_validation_for_llama_cpp(self, tmp_path) -> None:
        """Test that AnalysisConfig validates llama.cpp parameters."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()

        config = AnalysisConfig(
            backend="llama-cpp",
            model_path=str(model_file),
            n_gpu_layers=5,
            n_ctx=1024,
        )
        assert config.backend == "llama-cpp"
        assert config.n_gpu_layers == 5
        assert config.n_ctx == 1024

        with pytest.raises(ValueError, match="backend must be"):
            AnalysisConfig(backend="invalid")

    def test_config_backend_defaults(self, tmp_path) -> None:
        """Test default values for llama.cpp backend parameters."""
        model_file = tmp_path / "test.gguf"
        model_file.write_text("dummy")

        config = AnalysisConfig(backend="llama-cpp", model_path=str(model_file))
        assert config.n_gpu_layers == 0
        assert config.n_ctx == 2048
        assert config.n_threads is None

        config_auto = AnalysisConfig(backend="llama-cpp")
        assert config_auto.model_path is None
        assert config_auto.n_gpu_layers == 0
        assert config_auto.n_ctx == 2048
