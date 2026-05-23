"""Tests for shared device detection utility."""

from unittest.mock import MagicMock, patch

import pytest

from cordon.core.device import check_cuda_compatibility, detect_device


class TestDetectDevice:
    """Tests for detect_device function."""

    @patch("cordon.core.device.torch")
    def test_auto_detect_cpu_fallback(self, mock_torch: MagicMock) -> None:
        """Test auto-detection falls back to CPU when no GPU available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert detect_device(None) == "cpu"

    @patch("cordon.core.device.torch")
    def test_auto_detect_cuda(self, mock_torch: MagicMock) -> None:
        """Test auto-detection selects CUDA when available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            major=8, minor=0, name="RTX 3090"
        )
        assert detect_device(None) == "cuda"

    @patch("cordon.core.device.torch")
    def test_auto_detect_mps(self, mock_torch: MagicMock) -> None:
        """Test auto-detection selects MPS when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert detect_device(None) == "mps"

    @patch("cordon.core.device.torch")
    def test_explicit_cuda_unavailable_raises(self, mock_torch: MagicMock) -> None:
        """Test that explicitly requesting CUDA when unavailable raises RuntimeError."""
        mock_torch.cuda.is_available.return_value = False
        with pytest.raises(RuntimeError, match="CUDA device requested but CUDA is not available"):
            detect_device("cuda")

    @patch("cordon.core.device.torch")
    def test_explicit_cpu(self, mock_torch: MagicMock) -> None:
        """Test that explicitly requesting CPU works."""
        assert detect_device("cpu") == "cpu"

    @patch("cordon.core.device.torch")
    def test_explicit_mps(self, mock_torch: MagicMock) -> None:
        """Test that explicitly requesting MPS works."""
        assert detect_device("mps") == "mps"


class TestCheckCudaCompatibility:
    """Tests for check_cuda_compatibility function."""

    @patch("cordon.core.device.torch")
    def test_compatible_gpu(self, mock_torch: MagicMock) -> None:
        """Test that a compatible GPU passes without error."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            major=7, minor=5, name="RTX 2080"
        )
        check_cuda_compatibility()

    @patch("cordon.core.device.torch")
    def test_incompatible_gpu(self, mock_torch: MagicMock) -> None:
        """Test that an incompatible GPU raises RuntimeError."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            major=5, minor=0, name="GTX 960"
        )
        with pytest.raises(RuntimeError, match="GPU COMPATIBILITY ERROR"):
            check_cuda_compatibility()

    @patch("cordon.core.device.torch")
    def test_no_cuda_available(self, mock_torch: MagicMock) -> None:
        """Test that check passes when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        check_cuda_compatibility()
