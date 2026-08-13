"""Fail-fast provenance checks for the external NVIDIA userspace bundle."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Iterable


_NVIDIA_DRIVER_PREFIXES = (
    "libcuda.so",
    "libcudadebugger.so",
    "libEGL_nvidia.so",
    "libGLX_nvidia.so",
    "libGLESv1_CM_nvidia.so",
    "libGLESv2_nvidia.so",
    "libglxserver_nvidia.so",
    "libnvcuvid.so",
    "libnvidia-allocator.so",
    "libnvidia-api.so",
    "libnvidia-cfg.so",
    "libnvidia-egl-",
    "libnvidia-eglcore.so",
    "libnvidia-encode.so",
    "libnvidia-fbc.so",
    "libnvidia-glcore.so",
    "libnvidia-glsi.so",
    "libnvidia-glvkspirv.so",
    "libnvidia-gpucomp.so",
    "libnvidia-gtk2.so",
    "libnvidia-gtk3.so",
    "libnvidia-ml.so",
    "libnvidia-ngx.so",
    "libnvidia-nvvm.so",
    "libnvidia-opencl.so",
    "libnvidia-opticalflow.so",
    "libnvidia-pkcs11",
    "libnvidia-ptxjitcompiler.so",
    "libnvidia-rtcore.so",
    "libnvidia-sandboxutils.so",
    "libnvidia-tls.so",
    "libnvidia-vksc-core.so",
    "libnvidia-wayland-client.so",
    "libnvoptix.so",
    "libvdpau_nvidia.so",
    "nvidia_drv.so",
)

_REQUIRED_LOADED_PREFIXES = (
    "libcuda.so",
    "libEGL_nvidia.so",
    "libGLX_nvidia.so",
    "libnvidia-ml.so",
)


def loaded_nvidia_userspace_paths(map_lines: Iterable[str]) -> tuple[Path, ...]:
    """Return mapped NVIDIA driver libraries, excluding generic GLVND/CUDA runtimes."""

    paths: set[Path] = set()
    for line in map_lines:
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1])
        if path.name.startswith(_NVIDIA_DRIVER_PREFIXES):
            paths.add(path)
    return tuple(sorted(paths))


def audit_current_process(bundle_root: Path) -> dict[str, object]:
    """Prove that every currently mapped NVIDIA driver library belongs to the bundle."""

    expected_root = bundle_root.resolve()
    loaded = loaded_nvidia_userspace_paths(
        Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    )
    if not loaded:
        raise RuntimeError("No NVIDIA userspace driver libraries are mapped")

    loaded_names = tuple(path.name for path in loaded)
    missing = [
        prefix
        for prefix in _REQUIRED_LOADED_PREFIXES
        if not any(name.startswith(prefix) for name in loaded_names)
    ]
    if missing:
        raise RuntimeError(f"Core NVIDIA userspace libraries were not mapped: {missing}")

    outside = [path for path in loaded if not path.is_relative_to(expected_root)]
    if outside:
        rendered = ", ".join(str(path) for path in outside)
        raise RuntimeError(f"NVIDIA userspace libraries escaped the data-disk bundle: {rendered}")
    return {
        "bundle_root": str(expected_root),
        "all_nvidia_userspace_from_bundle": True,
        "mapped_driver_libraries": [str(path) for path in loaded],
    }


def load_and_check_cuda(bundle_root: Path) -> dict[str, object]:
    """Load the common driver entry points and perform a CUDA Driver API smoke check."""

    cuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libEGL_nvidia.so.0", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libGLX_nvidia.so.0", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    result = cuda.cuInit(0)
    if result != 0:
        raise RuntimeError(f"cuInit failed with CUDA driver error {result}")

    device_count = ctypes.c_int()
    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    result = cuda.cuDeviceGetCount(ctypes.byref(device_count))
    if result != 0:
        raise RuntimeError(f"cuDeviceGetCount failed with CUDA driver error {result}")

    audit = audit_current_process(bundle_root)
    audit["cuda_device_count"] = device_count.value
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args()
    audit = load_and_check_cuda(args.bundle_root)
    print(
        "UNIVTAC_NVIDIA_USERSPACE_AUDIT " + json.dumps(audit, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
