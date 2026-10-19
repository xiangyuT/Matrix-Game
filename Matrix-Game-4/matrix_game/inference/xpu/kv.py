"""Inference-only fixed-sink attention without full-window KV concatenation."""
import os

import torch

_status = {"overlay_calls": 0, "fallback_calls": 0}


def kv_backend_status():
    """Return process-local sink-overlay dispatch counts."""
    return dict(_status)


def _can_overlay(k_sink, cache, sink_tokens, recent_start, end, simple):
    k, v = cache["k"], cache["v"]
    return (simple and not torch.is_grad_enabled() and k.device.type == "xpu"
            and k.shape == v.shape and k.dtype == v.dtype and k.device == v.device
            and k.is_contiguous() and v.is_contiguous()
            and end == k.shape[1] and recent_start == sink_tokens
            and sink_tokens > 0 and k_sink.shape == k[:, :sink_tokens].shape
            and k_sink.dtype == k.dtype and k_sink.device == k.device)


def try_sink_attention(attention, query, k_sink, cache, sink_tokens,
                       recent_start, end, *, simple):
    """Return attention output, or None to retain the original concatenation.

    The caller has already committed the current KV and computed fixed-position
    sink keys. Only a full contiguous inference window with adjacent sink/recent
    slices and no memory/context prefix can borrow the cache itself. Restore the
    original sink even when the reader raises. All operations use the caller's
    current XPU stream, so restoration follows the attention reads in queue order.
    """
    enabled = os.environ.get("MATRIX_XPU_KV_SINK_OVERLAY", "1")
    if enabled not in ("0", "1"):
        raise ValueError("MATRIX_XPU_KV_SINK_OVERLAY must be 0 or 1")
    if enabled == "0" or not _can_overlay(
            k_sink, cache, sink_tokens, recent_start, end, simple):
        _status["fallback_calls"] += 1
        return None
    destination = cache["k"][:, :sink_tokens]
    saved = destination.clone()
    try:
        destination.copy_(k_sink)
        result = attention(query, cache["k"][:, :end], cache["v"][:, :end])
    finally:
        destination.copy_(saved)
    _status["overlay_calls"] += 1
    return result
