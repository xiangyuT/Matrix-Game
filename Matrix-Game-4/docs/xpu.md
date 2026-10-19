# Intel XPU development runtime

The XPU entry point uses the released 864x480 scene models, the full streaming
TAEHV codec, one denoising step, and the original 21-latent visual cache including
its identity sink. Text encoding, DiT linear/attention layers and the codec run
on XPU. CUDA runtimes and their dependency pins remain available separately.

This is a development integration. Correctness tests, real model output, browser
transport and steady-state performance are distinct checks; availability of this
entry point alone does not establish any of them.

## Environment

Use an Intel GPU container with a compatible Level Zero/compute runtime. The
initial environment is `intel/omix:0.4.0-devel-ubuntu24.04`; pin its full image
identity in local execution evidence. Select the intended GPU through a
runtime-supplied `ZE_AFFINITY_MASK` and verify Torch's UUID/PCI mapping before
device execution. No physical device ordinal is built into the integration.

Install PyTorch XPU and its matching torchvision first, then install the package
with `pip install -e '.[xpu,server,download,test]'`. The XPU extra accepts Torch
2.12 through 2.14, avoiding the CUDA extra's exact torch/torchvision pins; it is
not a claim that every accepted framework version has been device-tested.
Native game and GStreamer/MediaMTX dependencies are unchanged; see
[installation](installation.md). Model resources remain the pinned release
described in [resource distributions](../releases/README.md).

## Entry points

After model/resource installation, start MediaMTX with its supplied config, then:

```bash
bash scripts/serve.sh --scene ancient-egypt --resources /path/to/resources \
  --preset xpu --fps 20
```

The capture rate is a requested game cadence, not a measured model throughput.
Use the same preset for an existing clean low-poly recording:

```bash
matrix-game render --scene ancient-egypt --resources /path/to/resources \
  --preset xpu --input reference.mkv --output /path/to/new-output-directory
```

`--preset xpu` selects XPU, TAEHV, one-step inference, native DiT/T5 and no codec
graph. `--device xpu --preset quality` selects the original Wan/two-step route.
Explicit CUDA-only DiT, T5 or codec graphs are rejected rather than silently
replaced. Model runtime remains single-device. OpenGL device selection is
separate from Torch affinity and must be checked for the native game too. Set
`MATRIX_EGL_RENDER_NODE` to the verified GPU's DRM render-node path when multiple
GPUs are present. The native EGL renderer reports its chosen node and fails if
an explicitly selected node cannot initialize; without this variable it retains
the original first-available-device behavior.

## Precision and lifetime

The native runtime requires the XPU driver to expose FP64 support. It retains
the upstream FP64 RoPE, timestep sinusoid and scheduler arithmetic on XPU, with
the original final dtype conversions. Admission rejects devices without this
capability. No CPU offload or FP32 replacement for that arithmetic is implied
by this backend.

For fixed prompts, T5 is encoded on XPU and released before DiT loading. Codec
encoder and decoder feature caches, clean-history KV commits, sink positions and
output ownership follow the shared stream. XPU events and synchronization use
the selected accelerator API. Clean KV must still be committed even when RGB
is published early.

The Python-owned KV position indices stay in host int64 tensors to avoid
per-layer scalar device reads. Keys, values and model arithmetic remain on XPU.
Set `MATRIX_XPU_HOST_KV_METADATA=0` to retain device indices for development
comparisons. For bitwise rollout comparisons, enable Torch deterministic
algorithms on both sides; ordinary XPU execution can have BF16 run-to-run
variation that propagates through the rollout.

## Optional SYCL-TLA attention

The XPU runtime can use a separately built SYCL-TLA/CUTE sidecar. It keeps native
GEMM and the original codec, RoPE, scheduler and clean-KV pass. No ComfyUI or LLM
application framework is needed. Build inside the same oneAPI/Torch environment
that will run the model, using a clean Intel SYCL-TLA checkout at
`2fc09973bfdf15755090fcb0e3b6ad236408a992`:

```bash
source /opt/intel/oneapi/setvars.sh
python -m matrix_game.inference.xpu.build \
  --sycl-tla-root /path/to/sycl-tla \
  --output /path/to/new-build/matrix_tla.so
export MATRIX_XPU_TLA_LIBRARY=/path/to/new-build/matrix_tla.so
export MATRIX_XPU_ATTENTION=tla
```

The builder does not download sources. It validates pinned header hashes, applies
the BMG remainder-mask overlay with an exact-once check, and writes a library,
an adjacent `.so.json` manifest, and a `.so.build` directory containing commands
and logs. Existing output paths are rejected. Keep the library and its manifest
together. Torch version/revision, C++ ABI, target and binary hash are checked
before loading; rebuild after changing the framework environment.

`MATRIX_XPU_ATTENTION=auto` is the default: a configured compatible library enables
eligible TLA calls; without a library the runtime uses Torch SDPA. `sdpa` forces
the native route. `tla` requires a valid library when an eligible call is reached.
An incompatible library in `auto` warns once and falls back. Device execution
errors propagate. Use a fresh process to change libraries.

The current sidecar targets B70 (`0xE223`) inference with contiguous BF16 BLHD
tensors, batch one, head dimension 128, query length above 256 and at most 512,
and a full 21-query-block KV window. This describes the streaming-cache relation
without fixing a resolution or physical device ordinal. Early cache filling,
text cross-attention, other layouts/dtypes, causal/local-window requests,
custom scales, dropout, padding lengths and autograd retain native SDPA.
Performance evidence is specific to measured shapes; the structural admission
range is not a performance guarantee for every input.

TLA changes floating-point reductions. Compare full trajectories and generated
media against `MATRIX_XPU_ATTENTION=sdpa`; bitwise equality is not required for
this backend. `matrix_game.inference.xpu.backend_status()` reports actual TLA
and fallback call counts. The default Python dependency set does not build or
load this optional library implicitly.

## Checks

The ordinary suite runs without model weights. Device tests require an explicit
opt-in after a live mapping/idle check:

```bash
python -m unittest discover -s tests -p 'test_*.py'
MATRIX_RUN_XPU_TESTS=1 python -m unittest discover -s tests -p 'test_xpu.py'
```

The device cases cover actual 405-query attention with 405/512/8505-key inputs,
FP64-reference RoPE at early/window/long-session positions, and scheduler output.
They do not replace real scene generation, cache-rollover, stop/restart, media
decoding or browser checks in [validation](testing.md).
