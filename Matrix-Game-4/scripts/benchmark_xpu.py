#!/usr/bin/env python3
"""Reproduce native-stream XPU KV validation, development B/C/B, or a trace."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matrix_game.inference.loader import ModelComponents
from matrix_game.inference.preprocessing import frames_to_tensor
from matrix_game.inference.xpu import backend_status, kv_backend_status
from matrix_game.scenes import add_model_arguments, resolve_model_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--reference", type=Path, required=True, help="Recorded clean RGB uint8 [frames,H,W,3] .npy")
    parser.add_argument("--output", type=Path, required=True, help="New result directory, outside source distributions")
    parser.add_argument("--action", choices=("validate", "benchmark", "profile", "all"), default="validate")
    parser.add_argument("--warmup-chunks", type=int, default=25)
    parser.add_argument("--measured-chunks", type=int, default=16)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = resolve_model_arguments(parser.parse_args())
    if (args.device != "xpu" or args.prompt_mode != "fixed" or args.warmup_chunks < 25
            or args.measured_chunks < 3 or args.samples < 3):
        parser.error("requires an XPU device, fixed prompt, at least 25 warmup chunks, 3 measured chunks and 3 samples")
    if torch.xpu.device_count() != 1:
        parser.error("select and verify exactly one affinity-visible XPU before execution")
    frames = np.load(args.reference, mmap_mode="r", allow_pickle=False)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        parser.error("reference must contain uint8 RGB frames")
    count = args.warmup_chunks + args.measured_chunks
    if len(frames) < 1 + 4 * (count - 1):
        parser.error("reference is too short for the selected full chunks")
    blocks = [frames[:1]] + [frames[1 + 4 * i:5 + 4 * i] for i in range(count - 1)]
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    save("configuration.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "device": str(torch.xpu.get_device_properties(0)),
        "attention_policy": os.environ.get("MATRIX_XPU_ATTENTION", "auto"),
        "attention_library": os.environ.get("MATRIX_XPU_TLA_LIBRARY"),
        "evidence_class": "development_performance_comparison" if args.action in ("benchmark", "all") else "functional_or_trace",
        "scope": "CastleStream: preprocessing, codec, denoise, clean-KV and owned CPU RGB; excludes load, game, browser and transport",
        "text_mode": "cached fixed prompt; encoding outside timed region", "formal_acceptance": False})
    components = ModelComponents(args)
    components.load()
    def push(stream, i):
        return stream.push(frames_to_tensor(blocks[i]), frame_start=0 if i == 0 else 1 + 4 * (i - 1),
                           on_rgb=lambda rgb: None)
    @contextlib.contextmanager
    def route(enabled):
        old = os.environ.get("MATRIX_XPU_KV_SINK_OVERLAY")
        os.environ["MATRIX_XPU_KV_SINK_OVERLAY"] = "1" if enabled else "0"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("MATRIX_XPU_KV_SINK_OVERLAY", None)
            else:
                os.environ["MATRIX_XPU_KV_SINK_OVERLAY"] = old
    def counters():
        return {"attention": backend_status(), "kv": kv_backend_status()}
    actions = ("validate", "benchmark", "profile") if args.action == "all" else (args.action,)
    for action in actions:
        phase_output = args.output / action if args.action == "all" else args.output
        if args.action == "all":
            phase_output.mkdir()
        def save(name, value):
            (phase_output / name).write_text(json.dumps(value, indent=2) + "\n")
        if action == "validate":
            expected, cache = [], []
            with route(False):
                stream = components.stream(seed=args.seed)
                try:
                    for i in range(count):
                        result = push(stream, i)
                        expected.append((result.latent.cpu(), result.control.cpu(), result.rgb.clone()))
                    cache = [{k: v.detach().cpu().clone() for k, v in layer.items() if torch.is_tensor(v)}
                             for layer in stream.kv]
                finally:
                    stream.close()
            checks = []
            with route(True):
                stream = components.stream(seed=args.seed)
                try:
                    for i, (latent, control, rgb) in enumerate(expected):
                        result = push(stream, i)
                        row = {"chunk": i, "latent": torch.equal(result.latent.cpu(), latent),
                               "control": torch.equal(result.control.cpu(), control), "rgb": torch.equal(result.rgb, rgb)}
                        checks.append(row)
                        assert all(row[k] for k in ("latent", "control", "rgb")), row
                    matched = [{k: torch.equal(stream.kv[j][k].cpu(), value) for k, value in layer.items()}
                               for j, layer in enumerate(cache)]
                    assert all(all(layer.values()) for layer in matched), matched
                finally:
                    stream.close()
            assert kv_backend_status()["overlay_calls"] > 0, "eligible sink route was not exercised"
            save("validation.json", {"passed": True, "bitwise_chunks": checks, "all_layer_cache_fields": matched,
                                     "counters": counters(), "attention_policy_same_for_both": True})
            print("KV_VALIDATION_PASS", count, flush=True)
        elif action == "benchmark":
            rows = []
            for block, enabled in (("baseline_before", False), ("candidate", True), ("baseline_after", False)):
                with route(enabled):
                    for sample in range(args.samples):
                        stream = components.stream(seed=args.seed)
                        before = kv_backend_status()
                        chunks = []
                        try:
                            for i in range(count):
                                torch.xpu.synchronize()
                                start = time.perf_counter()
                                push(stream, i)
                                torch.xpu.synchronize()
                                chunks.append({"chunk": i, "frames": len(blocks[i]), "seconds": time.perf_counter() - start})
                        finally:
                            stream.close()
                        selected = chunks[args.warmup_chunks:]
                        seconds = sum(x["seconds"] for x in selected)
                        nframes = sum(x["frames"] for x in selected)
                        after = kv_backend_status()
                        rows.append({"block": block, "sample": sample, "frames": nframes, "seconds": seconds,
                                     "fps": nframes / seconds, "chunks": chunks,
                                     "kv_calls": {k: after[k] - before[k] for k in before}})
                        save("samples.json", rows)
                        print("SAMPLE", block, sample, rows[-1]["fps"], flush=True)
            times = {b: [x["seconds"] for x in rows if x["block"] == b]
                     for b in ("baseline_before", "candidate", "baseline_after")}
            paired = [(a + b) / 2 for a, b in zip(times["baseline_before"], times["baseline_after"])]
            changes = [c / b - 1 for b, c in zip(paired, times["candidate"])]
            drift = statistics.mean(times["baseline_after"]) / statistics.mean(times["baseline_before"]) - 1
            decision = {"baseline_fps": nframes / statistics.mean(paired),
                        "candidate_fps": nframes / statistics.mean(times["candidate"]),
                        "paired_mean_latency_change": statistics.mean(changes),
                        "per_pair_latency_changes": changes, "wins": sum(x < 0 for x in changes),
                        "baseline_drift": drift, "valid": abs(drift) <= .01,
                        "evidence_class": "development_performance_comparison", "formal_acceptance": False,
                        "counters": counters()}
            save("decision.json", decision)
            assert decision["valid"], "baseline drift exceeded development limit"
            print("DECISION", json.dumps(decision), flush=True)
        else:
            with route(True):
                stream = components.stream(seed=args.seed)
                try:
                    for i in range(args.warmup_chunks):
                        push(stream, i)
                    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
                    with torch.profiler.profile(activities=activities):
                        push(stream, args.warmup_chunks)
                    torch.xpu.synchronize()
                    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                        for i in range(args.warmup_chunks + 1, args.warmup_chunks + 3):
                            with torch.profiler.record_function("MG4/full_chunk"):
                                push(stream, i)
                        torch.xpu.synchronize()
                    prof.export_chrome_trace(str(phase_output / "trace.json"))
                    save("profile.json", {"counters": counters(), "profiled_chunks": 2})
                finally:
                    stream.close()
        if action == "validate":
            del expected, cache, checks, matched, result
            gc.collect()
    del components
    gc.collect()


if __name__ == "__main__":
    with torch.no_grad():
        main()
