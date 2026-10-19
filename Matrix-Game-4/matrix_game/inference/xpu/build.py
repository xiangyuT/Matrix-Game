"""Build optional B70 TLA attention from an explicitly supplied pinned source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_headers(root, destination, spec):
    root = Path(root).resolve(strict=True)
    # Trust only the explicitly supplied source path for these read-only queries;
    # a bind-mounted checkout may belong to a different UID inside a container.
    git = ["git", "-c", "safe.directory=" + str(root), "-C", str(root)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git + ["status", "--porcelain"], text=True)
    if revision != spec["source"]["revision"] or dirty:
        raise RuntimeError("SYCL-TLA must be a clean checkout at " + spec["source"]["revision"])
    for entry in spec["headers"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Pinned SYCL-TLA header changed: " + entry["path"])
    directory = destination / "flash_attention_v2/collective"
    directory.mkdir(parents=True)
    source = root / "applications/flash_attention_v2/collective"
    text = (source / "xe_fmha_fwd_mainloop.hpp").read_text()
    if text.count(spec["original"]) != 1:
        raise RuntimeError("BMG remainder overlay requires exactly one source match")
    (directory / "xe_fmha_fwd_mainloop.hpp").write_text(text.replace(spec["original"], spec["replacement"], 1))
    shutil.copyfile(source / "fmha_fusion.hpp", directory / "fmha_fusion.hpp")
    for entry in spec["overlay"]["files"]:
        if digest(destination / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Generated BMG overlay mismatch: " + entry["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sycl-tla-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="icpx")
    args = parser.parse_args()
    import torch
    from .attention import CONTRACT, TLA_REVISION

    source = Path(__file__).parent / "csrc"
    output = args.output.expanduser().resolve()
    metadata = output.with_suffix(output.suffix + ".json")
    build = output.with_suffix(output.suffix + ".build")
    if output.exists() or metadata.exists() or build.exists():
        raise RuntimeError("Use a new output path; existing build evidence is preserved")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise RuntimeError("Source the compatible oneAPI setvars.sh before building (icpx missing)")
    build.mkdir(parents=True)
    spec = json.loads((source / "provenance.json").read_text())
    prepare_headers(args.sycl_tla_root, build / "overlay", spec)
    tla = args.sycl_tla_root.resolve()
    torch_root = Path(torch.__file__).parent
    command = [compiler, "-fsycl", "-std=c++20", "-O3", "-DNDEBUG", "-fPIC", "-shared",
               "-fsycl-targets=spir64_gen", "-Xsycl-target-backend", "-device bmg_g31",
               "-Xspirv-translator", "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
               "-fno-sycl-instrument-device-code", "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET",
               "-DOMNI_XPU_ARCH_BMG", "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi())),
               "-DCUTE_TUNE_Q_TILE=256", "-DCUTE_TUNE_KV_TILE=32", "-DCUTE_TUNE_V_TILE=32",
               "-DCUTE_TUNE_MMA_K=32", "-DCUTE_TUNE_SUBGROUP_LAYOUT_Q=16",
               "-DCUTE_TUNE_PIPELINE_STAGES=2", "-DCUTE_TUNE_GRF_SIZE=256"]
    includes = [build / "overlay", source, tla / "include", tla / "tools/util/include",
                tla / "examples/common", tla / "applications", tla / "applications/flash_attention_v2/collective",
                Path(sysconfig.get_path("include")), torch_root / "include", torch_root / "include/torch/csrc/api/include"]
    command.extend("-I" + str(path) for path in includes)
    command.extend(["-Wno-deprecated-declarations", "-Wno-unused-variable", "-L" + str(torch_root / "lib"),
                    "-ltorch", "-ltorch_python", "-ltorch_cpu", "-ltorch_xpu", "-lc10", "-lc10_xpu",
                    "-Wl,-rpath," + str(torch_root / "lib"), "-o", str(output), str(source / "attention.cpp")])
    record = {"contract": CONTRACT, "torch": str(torch.__version__), "torch_git": torch.version.git_version,
              "cxx11_abi": bool(torch.compiled_with_cxx11_abi()), "target": "bmg_g31",
              "sycl_tla_revision": TLA_REVISION,
              "compiler": subprocess.check_output([compiler, "--version"], text=True),
              "source_sha256": {p.name: digest(p) for p in source.iterdir() if p.is_file()},
              "command": command}
    (build / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    with (build / "build.log").open("w") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    record.update(returncode=run.returncode, seconds=time.monotonic() - start)
    (build / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    if run.returncode:
        raise RuntimeError("TLA build failed; see " + str(build / "build.log"))
    record["sha256"] = digest(output)
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"library": str(output), "manifest": str(metadata), "sha256": record["sha256"]}))


if __name__ == "__main__":
    main()
