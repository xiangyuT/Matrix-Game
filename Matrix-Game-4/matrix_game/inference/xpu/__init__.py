"""Optional Intel XPU attention; native Torch SDPA remains the fallback."""
from .attention import backend_status, try_attention
from .kv import kv_backend_status

__all__ = ["backend_status", "try_attention", "kv_backend_status"]
