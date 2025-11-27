"""Device utility for checking CUDA availability and selecting appropriate device."""

import logging

logger = logging.getLogger(__name__)


def get_available_device(requested_device: str | None = None) -> str:
    """
    Get available device, falling back to CPU if CUDA is not available.

    Args:
        requested_device: Requested device ("cuda", "cpu", or None for auto-detect)

    Returns:
        Available device string ("cuda" or "cpu")

    Examples:
        >>> get_available_device("cuda")  # Falls back to CPU if CUDA unavailable
        'cpu'
        >>> get_available_device("cpu")   # Always works
        'cpu'
        >>> get_available_device(None)    # Auto-detect
        'cuda' # or 'cpu' depending on availability
    """
    if requested_device is None or requested_device.lower() == "auto":
        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            return "cpu"

    requested_device = requested_device.lower()

    if requested_device == "cpu":
        # CPU always works
        return "cpu"

    if requested_device == "cuda":
        # Check if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")
            return "cpu"

    # Unknown device, fall back to CPU
    logger.warning(f"Unknown device '{requested_device}', falling back to CPU")
    return "cpu"
