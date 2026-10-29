"""Verify the CUDA/cuDNN stack a JAX training process will actually load.

Checking cuda_versions.cudnn_get_version() alone is NOT sufficient: the number it
returns comes from the cuDNN sub-libraries, so a 9.10.2 dispatcher stub driving
9.14 engine libraries still reports 91400. This walks /proc/self/maps to check
the loaded library sources and versions. It does not validate training numerics.

Run it in the SAME shell that launches training:
    python scripts/check_cuda_stack.py
"""

import collections
import importlib.metadata
import os
import re
import sys

FAIL = []
EXPECTED_CUDA = "12.8"
# The three torch wheels must come from the cu128 index. A CPU-only wheel installs cleanly
# and silently removes everything this script is here to check.
CU_SUFFIX = "+cu128"
CU_WHEELS = ("torch", "torchvision", "torchcodec")


def major_minor(version):
    """'2.10.0+cu128' -> '2.10'. Also handles '9.10.2.21' and '12.8'."""
    return ".".join(re.split(r"[.+]", str(version))[:2])


def loaded_libs(pattern):
    out = collections.OrderedDict()
    with open("/proc/self/maps") as f:
        for line in f:
            m = re.search(r"(/\S*" + pattern + r"\S*)", line)
            if m:
                out[m.group(1)] = None
    return list(out)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def report(name, pattern):
    libs = loaded_libs(pattern)
    if not libs:
        print(f"  {name}: (none loaded)")
        return
    dirs = {os.path.dirname(p) for p in libs}
    for p in libs:
        print(f"    {p}")
    if len(dirs) > 1:
        FAIL.append(f"{name} is loaded from {len(dirs)} different directories: {sorted(dirs)}")
        print(f"  !! {name}: MIXED — {len(dirs)} source directories")
    else:
        print(f"  OK {name}: single source {dirs.pop()}")


def main():
    import jax
    from jax._src.lib import cuda_versions
    import jax.numpy as jnp
    import torch
    import torchcodec
    import torchvision

    section("1. reported versions")
    rt, build = cuda_versions.cudnn_get_version(), cuda_versions.cudnn_build_version()
    cudnn_package_version = importlib.metadata.version("nvidia-cudnn-cu12")
    torch_rt = torch.backends.cudnn.version()
    print(f"  jax                    : {jax.__version__}")
    print(f"  torch                  : {torch.__version__}")
    print(f"  torchvision            : {torchvision.__version__}")
    print(f"  torchcodec             : {torchcodec.__version__}")
    print(f"  nvidia-cudnn-cu12      : {cudnn_package_version}")
    print(f"  torch CUDA             : {torch.version.cuda}")
    print(f"  torch cuDNN runtime    : {torch_rt}")
    print(f"  cuDNN runtime (reported): {rt}")
    print(f"  cuDNN build            : {build}")
    print(f"  LD_LIBRARY_PATH        : {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")

    actual = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchcodec": torchcodec.__version__,
        "nvidia-cudnn-cu12": cudnn_package_version,
        "cuda": torch.version.cuda,
    }
    if major_minor(actual["cuda"]) != EXPECTED_CUDA:
        FAIL.append(f"torch must be built against CUDA {EXPECTED_CUDA}, got {actual['cuda']}")
    FAIL.extend(
        f"{name} must be a {CU_SUFFIX} wheel, got {actual[name]}"
        for name in CU_WHEELS
        if not str(actual[name]).endswith(CU_SUFFIX)
    )
    if torch_rt != rt:
        FAIL.append(f"PyTorch reports cuDNN {torch_rt}, while JAX reports {rt}")

    # Force the CUDA libraries to actually load before inspecting the memory map.
    dev = jax.devices()[0]
    jnp.ones((8, 8)).block_until_ready()
    print(f"  device                 : {dev} (compute {dev.compute_capability})")

    section("2. libraries actually mapped into this process")
    report("libcudnn", "libcudnn")
    report("libcublas", "libcublas")
    report("libnccl", "libnccl")

    # A version embedded in a filename that disagrees with the reported runtime is
    # the exact failure that produced a 9.10.2 dispatcher over 9.14 engines.
    vers = set()
    for p in loaded_libs("libcudnn"):
        m = re.search(r"\.so\.(\d+)\.(\d+)\.(\d+)", p)
        if m:
            vers.add(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
    if len(vers) > 1:
        FAIL.append(f"cuDNN filenames span multiple versions: {sorted(vers)}")
        print(f"  !! cuDNN file versions: MIXED {sorted(vers)}")
    elif vers:
        v = vers.pop()
        maj, minor, patch = (int(x) for x in v.split("."))
        want = maj * 10000 + minor * 100 + patch
        print(f"  cuDNN file version     : {v}")
        if want != rt:
            FAIL.append(f"cuDNN filename says {v} ({want}) but runtime reports {rt}")
            print(f"  !! filename {want} != reported runtime {rt}")

    section("verdict")
    if FAIL:
        print("  FAIL — do not start production training:")
        for f in FAIL:
            print(f"    - {f}")
        return 1
    print("  PASS — CUDA stack checks passed: loaded library sources and versions are consistent.")
    print("  NOTE: library checks do not prove training is numerically stable or converges.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
