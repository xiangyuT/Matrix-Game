"""Guarded, inference-only SYCL-TLA dispatch for a full 21-latent window."""
import hashlib
import json
import os
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import torch

CONTRACT = "bmg-bf16-blhd-d128-q256-window21-v1"
TLA_REVISION = "2fc09973bfdf15755090fcb0e3b6ad236408a992"
_lock = threading.Lock()
_loaded_path = None
_status = {"tla_calls": 0, "sdpa_calls": 0, "library": None, "load_error": None}


def backend_status():
    """Return dispatch counters without initializing an accelerator."""
    return dict(_status)


def _supports_layout(q, k, v, *, q_lens=None, k_lens=None, dropout_p=0.,
                     softmax_scale=None, q_scale=None, causal=False,
                     window_size=(-1, -1)):
    if (q_lens is not None or k_lens is not None or dropout_p != 0.
            or softmax_scale is not None or q_scale is not None or causal
            or window_size != (-1, -1)):
        return False
    if any(t.ndim != 4 or t.dtype != torch.bfloat16 for t in (q, k, v)):
        return False
    if k.device != q.device or v.device != q.device:
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return False
    b, length, heads, dim = q.shape
    return (b == 1 and 256 < length <= 512 and heads > 0 and dim == 128
            and k.shape == (b, 21 * length, heads, dim) and v.shape == k.shape
            and all(t.is_contiguous() for t in (q, k, v)))


@lru_cache(maxsize=None)
def _supported_device(index):
    return torch.xpu.get_device_properties(index).device_id == 0xE223


@lru_cache(maxsize=None)
def _load_library(path):
    global _loaded_path
    library = Path(path).expanduser().resolve(strict=True)
    metadata = json.loads(library.with_suffix(library.suffix + ".json").read_text())
    if not isinstance(metadata, dict):
        raise RuntimeError("TLA build manifest must be a JSON object")
    expected = {"contract": CONTRACT, "torch": str(torch.__version__),
                "torch_git": torch.version.git_version,
                "cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
                "sycl_tla_revision": TLA_REVISION, "target": "bmg_g31"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"TLA library {key} mismatch; rebuild with this Torch environment")
    if hashlib.sha256(library.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise RuntimeError("TLA library hash does not match its build manifest")
    with _lock:
        if _loaded_path is not None and _loaded_path != library:
            raise RuntimeError("Use a fresh process to change the TLA library")
        if _loaded_path is None:
            if hasattr(torch.ops.matrix_xpu_tla, "sdp"):
                raise RuntimeError("TLA operator already registered outside this loader; use a fresh process")
            torch.ops.load_library(str(library))
            _loaded_path = library
            _status["library"] = str(library)
    return torch.ops.matrix_xpu_tla.sdp


@lru_cache(maxsize=None)
def _resolve_library(path, strict):
    try:
        return _load_library(path)
    except (OSError, ValueError, RuntimeError) as error:
        _status["load_error"] = str(error)
        if strict:
            raise RuntimeError(f"Cannot load Matrix4 TLA attention: {error}") from error
        warnings.warn(f"Matrix4 TLA attention unavailable; using Torch SDPA: {error}", RuntimeWarning)
        return None


def try_attention(q, k, v, **options):
    """Return TLA output, or None when the caller should use its native route.

    auto uses an explicitly configured library; sdpa disables it; tla requires
    that library for eligible calls. Unsupported semantics always use SDPA.
    Device execution errors propagate, rather than retrying a possibly bad queue.
    """
    if q.device.type != "xpu":
        return None
    mode = os.environ.get("MATRIX_XPU_ATTENTION", "auto")
    if mode not in ("auto", "sdpa", "tla"):
        raise ValueError("MATRIX_XPU_ATTENTION must be auto, sdpa or tla")
    path = os.environ.get("MATRIX_XPU_TLA_LIBRARY")
    if mode == "sdpa" or (mode == "auto" and not path):
        _status["sdpa_calls"] += 1
        return None
    if not _supports_layout(q, k, v, **options) or not _supported_device(q.device.index):
        _status["sdpa_calls"] += 1
        return None
    if not path:
        if mode == "tla":
            raise RuntimeError("Set MATRIX_XPU_TLA_LIBRARY to the output of matrix_game.inference.xpu.build")
        _status["sdpa_calls"] += 1
        return None
    op = _resolve_library(path, mode == "tla")
    if op is None:
        _status["sdpa_calls"] += 1
        return None
    result = op(q, k, v)
    _status["tla_calls"] += 1
    return result
