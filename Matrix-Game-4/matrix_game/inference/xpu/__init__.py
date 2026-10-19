"""Optional Intel XPU attention; native Torch SDPA remains the fallback."""
from .attention import backend_status, try_attention

__all__ = ["backend_status", "try_attention"]
