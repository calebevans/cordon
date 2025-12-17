from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from litellm.exceptions import AuthenticationError, RateLimitError, Timeout

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class TestRemoteVectorizerConfiguration:
    """Tests for RemoteVectorizer configuration and initialization."""

    def test_remote_backend_creates_remote_vectorizer(self) -> None:
        """Test that remote backend creates RemoteVectorizer."""
        from cordon.embedding import create_vectorizer

        config = AnalysisConfig(backend="remote", model_name="openai/text-embedding-3-small")
        vectorizer = create_vectorizer(config)

        from cordon.embedding.remote import RemoteVectorizer

        assert isinstance(vectorizer, RemoteVectorizer)

    def test_config_validation_for_remote(self) -> None:
        """Test that AnalysisConfig validates remote backend."""
        config = AnalysisConfig(
            backend="remote",
            model_name="openai/text-embedding-3-small",
            api_key="test-key",
            endpoint="https://api.example.com",
            request_timeout=30.0,
        )
        assert config.backend == "remote"
        assert config.api_key == "test-key"
        assert config.endpoint == "https://api.example.com"
        assert config.request_timeout == 30.0

    def test_invalid_timeout_raises_error(self) -> None:
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="request_timeout must be > 0"):
            AnalysisConfig(backend="remote", request_timeout=0)

        with pytest.raises(ValueError, match="request_timeout must be > 0"):
            AnalysisConfig(backend="remote", request_timeout=-1)


class TestRemoteVectorizerEmbedding:
    """Tests for RemoteVectorizer embedding functionality."""

    @pytest.fixture
    def mock_litellm_response(self):
        """Create a mock response from litellm.embedding()."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
        ]
        return mock_response

    @pytest.fixture
    def vectorizer(self):
        """Create a RemoteVectorizer instance for testing."""
        from cordon.embedding.remote import RemoteVectorizer

        config = AnalysisConfig(
            backend="remote",
            model_name="openai/text-embedding-3-small",
            api_key="test-key",
            batch_size=2,
        )
        return RemoteVectorizer(config)

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_embed_single_window(self, mock_embedding, vectorizer, mock_litellm_response) -> None:
        """Test embedding a single text window."""
        mock_embedding.return_value = mock_litellm_response

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
        assert embedding.shape[0] == 4

        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["model"] == "openai/text-embedding-3-small"
        assert call_kwargs["input"] == ["Error: Connection timeout"]
        assert call_kwargs["api_key"] == "test-key"

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_embedding_normalization(self, mock_embedding, vectorizer) -> None:
        """Test that embeddings are L2 normalized."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [3.0, 4.0]},
        ]
        mock_embedding.return_value = mock_response

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

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_empty_windows_list(self, mock_embedding, vectorizer) -> None:
        """Test embedding empty list of windows."""
        results = list(vectorizer.embed_windows([]))
        assert len(results) == 0
        mock_embedding.assert_not_called()

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_batching(self, mock_embedding, vectorizer) -> None:
        """Test that batching works correctly."""

        def mock_embedding_response(**kwargs):
            """Return embeddings matching the number of input texts."""
            input_texts = kwargs.get("input", [])
            mock_response = MagicMock()
            mock_response.data = [
                {"embedding": [0.1 * (i + 1), 0.2 * (i + 1)]} for i in range(len(input_texts))
            ]
            return mock_response

        mock_embedding.side_effect = mock_embedding_response

        windows = [
            TextWindow(content=f"Log line {i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 6)
        ]

        results = list(vectorizer.embed_windows(windows))

        assert len(results) == 5
        assert mock_embedding.call_count == 3

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_authentication_error(self, mock_embedding, vectorizer) -> None:
        """Test handling of authentication errors."""
        mock_embedding.side_effect = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="text-embedding-3-small",
        )

        window = TextWindow(
            content="Test content",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        with pytest.raises(RuntimeError, match="Authentication failed"):
            list(vectorizer.embed_windows([window]))

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_rate_limit_error(self, mock_embedding, vectorizer) -> None:
        """Test handling of rate limit errors."""
        mock_embedding.side_effect = RateLimitError(
            message="Too many requests",
            llm_provider="openai",
            model="text-embedding-3-small",
        )

        window = TextWindow(
            content="Test content",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            list(vectorizer.embed_windows([window]))

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_timeout_error(self, mock_embedding, vectorizer) -> None:
        """Test handling of timeout errors."""
        mock_embedding.side_effect = Timeout(
            message="Request timeout after 60 seconds",
            llm_provider="openai",
            model="text-embedding-3-small",
        )

        window = TextWindow(
            content="Test content",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        with pytest.raises(RuntimeError, match="Request timeout"):
            list(vectorizer.embed_windows([window]))

    @patch("cordon.embedding.remote.litellm.embedding")
    def test_generic_error(self, mock_embedding, vectorizer) -> None:
        """Test handling of generic errors."""
        mock_embedding.side_effect = Exception("Some unexpected error")

        window = TextWindow(
            content="Test content",
            start_line=1,
            end_line=1,
            window_id=0,
        )

        with pytest.raises(RuntimeError, match="Error calling remote embedding API"):
            list(vectorizer.embed_windows([window]))
