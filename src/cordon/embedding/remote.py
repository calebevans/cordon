from collections.abc import Iterable, Iterator
from typing import Any

import litellm
import numpy as np
import numpy.typing as npt
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    Timeout,
)
from tqdm import tqdm

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow
from cordon.embedding.normalize import normalize_embeddings


class RemoteEmbedder:
    """Convert text windows to embeddings using remote API providers via LiteLLM."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the embedder with remote API configuration.

        Args:
            config: Analysis configuration with remote backend settings.
        """
        self.config = config
        litellm.set_verbose = False

    def embed_windows(
        self, windows: Iterable[TextWindow]
    ) -> Iterator[tuple[TextWindow, npt.NDArray[np.floating[Any]]]]:
        """Embed text windows into vector representations using remote API.

        Args:
            windows: Iterable of text windows to embed.

        Yields:
            Tuples of (window, embedding) where embeddings are L2-normalized.
        """
        window_list = list(windows)

        if not window_list:
            return

        batch_size = self.config.batch_size
        total_batches = (len(window_list) + batch_size - 1) // batch_size

        for batch_start_idx in tqdm(
            range(0, len(window_list), batch_size),
            desc="Generating embeddings",
            total=total_batches,
            unit="batch",
        ):
            batch = window_list[batch_start_idx : batch_start_idx + batch_size]
            texts = [window.content for window in batch]

            batch_results: list[tuple[TextWindow, npt.NDArray[np.floating[Any]]]] = []
            try:
                response = litellm.embedding(
                    model=self.config.model_name,
                    input=texts,
                    api_key=self.config.api_key,
                    api_base=self.config.endpoint,
                    timeout=self.config.request_timeout,
                )

                embeddings_data = response.data

                try:
                    raw_embeddings = np.array(
                        [item["embedding"] for item in embeddings_data],
                        dtype=np.float32,
                    )
                except (KeyError, TypeError) as error:
                    raise RuntimeError(
                        f"Unexpected API response format from model '{self.config.model_name}'. "
                        f"Expected 'embedding' field in response data: {error}"
                    ) from error

                normalized = normalize_embeddings(raw_embeddings)
                batch_results = list(zip(batch, normalized, strict=True))

            except AuthenticationError as error:
                raise RuntimeError(
                    f"Authentication failed for model '{self.config.model_name}'. "
                    f"Please check your API key or environment variables. Error: {error}"
                ) from error
            except RateLimitError as error:
                raise RuntimeError(
                    f"Rate limit exceeded for model '{self.config.model_name}'. "
                    f"Try again later or reduce batch_size. Error: {error}"
                ) from error
            except Timeout as error:
                raise RuntimeError(
                    f"Request timeout for model '{self.config.model_name}'. "
                    f"Try increasing request_timeout (current: {self.config.request_timeout}s). "
                    f"Error: {error}"
                ) from error
            except RuntimeError:
                raise
            except Exception as error:
                raise RuntimeError(
                    f"Error calling remote embedding API for model "
                    f"'{self.config.model_name}': {error}"
                ) from error

            yield from batch_results
