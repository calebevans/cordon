"""Shared device detection and CUDA compatibility utilities."""

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)


def detect_device(requested: Literal["cuda", "mps", "cpu"] | None = None) -> str:
    """Detect the best available device for PyTorch operations.

    Args:
        requested: Explicitly requested device, or None for auto-detection.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.

    Raises:
        RuntimeError: If the requested device is unavailable or incompatible.
    """
    if requested is not None:
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Use --device cpu or --device mps, or install a CUDA-compatible PyTorch build."
                )
            check_cuda_compatibility()
        elif requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but MPS is not available. "
                    "Use --device cpu or --device cuda."
                )
        return requested

    # auto-detect: cuda > mps > cpu
    if torch.cuda.is_available():
        check_cuda_compatibility()
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_cuda_compatibility() -> None:
    """Check if CUDA GPU is compatible with current PyTorch build.

    Raises:
        RuntimeError: If GPU compute capability is too old for PyTorch 2.0+.
    """

    if not torch.cuda.is_available():
        return

    device_props = torch.cuda.get_device_properties(0)

    if device_props.major < 6:
        gpu_name = device_props.name
        compute_capability = f"{device_props.major}.{device_props.minor}"
        raise RuntimeError(
            f"\n{'=' * 70}\n"
            f"GPU COMPATIBILITY ERROR\n"
            f"{'=' * 70}\n"
            f"GPU: {gpu_name}\n"
            f"Compute Capability: {compute_capability}\n"
            f"\n"
            f"PyTorch 2.0+ requires compute capability >= 6.0 (Pascal architecture or newer).\n"
            f"Your GPU has compute capability {compute_capability}, which is not supported.\n"
            f"\n"
            f"Options:\n"
            f"1. Use CPU mode: --device cpu\n"
            f"2. Use a newer GPU (Pascal/GTX 10-series or later)\n"
            f"3. Use llama.cpp backend for CPU inference:\n"
            f"   cordon --backend llama-cpp --n-threads 8 <file>\n"
            f"\n"
            f"Supported GPUs: GTX 10-series, RTX series, Tesla P/V/A series, or newer\n"
            f"{'=' * 70}\n"
        )
