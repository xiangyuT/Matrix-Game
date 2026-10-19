# XPU development measurements

Measured 2026-10-19 on Arc Pro B70 32 GB (`0xE223`), using the fixed-sink
integration on parent source `229542fa340fa786c29abb99fb305a23f34f6e16`.
These are development measurements, not formal release or browser acceptance.

## Compared routes

Both routes use native BF16 GEMM and the same optional SYCL-TLA attention.
The baseline disables `MATRIX_XPU_KV_SINK_OVERLAY`; the candidate enables it.
No quantization, QKV weight fusion, codec substitution between routes, omitted
clean pass, or changed attention arithmetic is part of this comparison.

| Route | Pipeline throughput |
| --- | ---: |
| Original KV concatenation + TLA | 17.170 FPS |
| Integrated sink overlay + TLA | 20.633 FPS |

Throughput improved 20.17%; paired mean latency changed
-16.79%, with 3/3 wins. Before/after
baseline latency drift was 0.483%.

| Block | Sample 0 FPS | Sample 1 FPS | Sample 2 FPS |
| --- | ---: | ---: | ---: |
| baseline_before | 17.2045 | 17.2236 | 17.2054 |
| candidate | 20.6667 | 20.5854 | 20.6472 |
| baseline_after | 17.1467 | 17.1016 | 17.1367 |

## Measurement boundary

- Released Ancient Egypt model, 864x480, TAEHV, one denoising pass and a separate
  clean-history pass per chunk; original FP64 RoPE/scheduler and 21-latent cache.
- Seed 0, deterministic algorithms, cached fixed prompt. Text encoding and model
  loading are outside the timed region and remain on XPU.
- Three fresh streams per baseline-before/candidate/baseline-after block. Each
  consumes 161 recorded frames: 25 warmup chunks are excluded; 16 chunks / 64
  frames are measured. The first chunk has one frame, later chunks have four.
- Timing includes preprocessing, codec, denoise, clean-KV and owned CPU RGB
  publication with XPU synchronization at chunk boundaries. It excludes game
  rendering, browser/network transport and action-to-display latency.
- Reference recording SHA256:
  `6b71840b4adbfdb4e8f6eaf44c7a246ccc8dc21fa0254d83520fae32beee82bc`. The recording is not part of the source package;
  throughput depends on its content and the selected runtime configuration.

## Correctness and environment

With attention held fixed, enabling the KV overlay preserved all 41 chunks of
latent/control/RGB and all 30 layers' final K/V, raw sink and int64 metadata
bitwise. Captured first-layer checks also match the previous measured prototype
at prefill, window edge, rolling, fixed-sink, long-position and grown-cache
boundaries. Reader-exception checks preserve final cache contents. This is a
lossless KV-dataflow comparison; TLA itself changes floating-point reductions
relative to native SDPA.

- OMIX `intel/omix:0.4.0-devel-ubuntu24.04`, image ID
  `sha256:53f0003fa66044935d69065836ba8ac40fab289b94655667fdeac0bdfa7e75c6`.
- Torch `2.14.0+xpu`, source
  `08187d9e0fba026dc8217405802ab5381dc88d90`; driver `1.17.39395+13`.
- Native GEMM retained. TLA uses the pinned source/build contract documented in
  [XPU setup](xpu.md#optional-sycl-tla-attention).

Use [the reproduction commands](xpu.md#reproduce-kv-validation-and-throughput)
with `scripts/benchmark_xpu.py`. Its JSON outputs include every measured chunk,
paired deltas, baseline drift and dispatch counters. Keep attention policy,
model, reference, precision, codec, prompt and timing boundary identical when
comparing KV routes. Formal image/browser/long-session acceptance is separate.
