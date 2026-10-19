"""Explicit accelerator selection shared by model loading and stream lifecycle."""
import os
import torch


def accelerator(device):
    kind = torch.device(device).type
    if kind == "cpu":
        return None
    if kind not in ("cuda", "xpu"):
        raise ValueError(f"Unsupported inference device: {kind}")
    return getattr(torch, kind)


def inference_device(name):
    device = torch.device(name)
    api = accelerator(device)
    if api is None or not api.is_available():
        raise RuntimeError(f"Requested inference accelerator is unavailable: {name}")
    if device.type == "xpu" and not api.get_device_properties(device).has_fp64:
        raise RuntimeError("The native XPU runtime requires FP64 support for the original RoPE and scheduler")
    return device


def empty_cache(device):
    api = accelerator(device)
    if api is not None:
        api.empty_cache()


def prepare_kv_metadata(caches, device):
    """Keep Python-owned inference indices on the host, avoiding scalar XPU reads."""
    if torch.device(device).type != "xpu":
        return
    enabled = os.environ.get("MATRIX_XPU_HOST_KV_METADATA", "1")
    if enabled not in ("0", "1"):
        raise ValueError("MATRIX_XPU_HOST_KV_METADATA must be 0 or 1")
    if enabled == "1":
        for cache in caches:
            for name in ("global_end_index", "local_end_index"):
                cache[name] = cache[name].cpu()
