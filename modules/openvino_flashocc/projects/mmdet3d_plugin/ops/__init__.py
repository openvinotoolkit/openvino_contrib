"""Operations with CPU fallback support."""
from .bev_pool import bev_pool, bev_pool_v2
from .nearest_assign import nearest_assign

try:
    from .bev_pool_v2 import TRTBEVPoolv2
except Exception:
    class TRTBEVPoolv2:
        """Placeholder for TensorRT BEV pool when TRT/CUDA is unavailable."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "TRTBEVPoolv2 requires TensorRT/CUDA and is not available in this environment. "
                "Use bev_pool_v2 instead."
            )


__all__ = ["bev_pool", "bev_pool_v2", "TRTBEVPoolv2", "nearest_assign"]
