from pathlib import Path

from agent_env.nvidia_runtime import loaded_nvidia_userspace_paths


def test_loaded_nvidia_paths_exclude_generic_loaders_and_cuda_runtime() -> None:
    lines = [
        "7f00-7f01 r-xp 0 00:00 0 /bundle/libcuda.so.570.124.06",
        "7f01-7f02 r-xp 0 00:00 0 /bundle/libnvidia-glcore.so.570.124.06",
        "7f02-7f03 r-xp 0 00:00 0 /conda/libEGL.so.1.1.0",
        "7f03-7f04 r-xp 0 00:00 0 /conda/libcudart.so.12",
        "7f04-7f05 r-xp 0 00:00 0 /isaac/libnvidia-ngx-dlss.so.3.7.10",
    ]
    assert loaded_nvidia_userspace_paths(lines) == (
        Path("/bundle/libcuda.so.570.124.06"),
        Path("/bundle/libnvidia-glcore.so.570.124.06"),
    )
