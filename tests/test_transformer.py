"""Unit tests for sentence-transformers vectorizer backend."""

import numpy as np
import pytest

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class TestTransformerVectorizerConfiguration:
    """Tests for TransformerVectorizer configuration and initialization."""

    def test_default_backend_creates_transformer(self) -> None:
        """Test that default backend creates TransformerVectorizer."""
        from cordon.embedding import create_vectorizer

        config = AnalysisConfig()
        vectorizer = create_vectorizer(config)

        from cordon.embedding.transformer import TransformerVectorizer

        assert isinstance(vectorizer, TransformerVectorizer)

    def test_explicit_sentence_transformers_backend(self) -> None:
        """Test explicit sentence-transformers backend selection."""
        from cordon.embedding import create_vectorizer

        config = AnalysisConfig(backend="sentence-transformers")
        vectorizer = create_vectorizer(config)

        from cordon.embedding.transformer import TransformerVectorizer

        assert isinstance(vectorizer, TransformerVectorizer)

    def test_device_detection(self) -> None:
        """Test device detection logic."""
        from cordon.embedding.transformer import TransformerVectorizer

        config = AnalysisConfig(device="cpu")
        vectorizer = TransformerVectorizer(config)

        assert vectorizer.device == "cpu"

    def test_custom_model_name(self) -> None:
        """Test custom model name configuration."""
        from cordon.embedding.transformer import TransformerVectorizer

        config = AnalysisConfig(model_name="all-MiniLM-L6-v2")
        vectorizer = TransformerVectorizer(config)

        assert vectorizer.config.model_name == "all-MiniLM-L6-v2"


class TestTransformerVectorizerEmbedding:
    """Tests for TransformerVectorizer embedding functionality."""

    @pytest.fixture
    def vectorizer(self):
        """Create a TransformerVectorizer instance for testing."""
        from cordon.embedding.transformer import TransformerVectorizer

        config = AnalysisConfig(device="cpu", batch_size=2)
        return TransformerVectorizer(config)

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

    def test_batching(self, vectorizer) -> None:
        """Test that batching works correctly."""
        windows = [
            TextWindow(content=f"Log line {i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 6)
        ]

        results = list(vectorizer.embed_windows(windows))

        assert len(results) == 5
        for i, (window, embedding) in enumerate(results):
            assert window.window_id == i
            assert window.start_line == i + 1
            assert embedding.shape[0] > 0


class TestTransformerVectorizerIntegration:
    """Integration tests with the full analysis pipeline."""

    def test_factory_creates_transformer_by_default(self) -> None:
        """Test that factory creates TransformerVectorizer by default."""
        from cordon.embedding import create_vectorizer

        config = AnalysisConfig()
        vectorizer = create_vectorizer(config)

        from cordon.embedding.transformer import TransformerVectorizer

        assert isinstance(vectorizer, TransformerVectorizer)
        assert vectorizer.config.backend == "sentence-transformers"
