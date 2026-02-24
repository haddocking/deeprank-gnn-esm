import shutil
import subprocess

import pytest
import torch


def _gpu_physically_present() -> bool:
    """Check if an NVIDIA GPU is present via nvidia-smi, independent of torch."""
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def pytest_collection_modifyitems(config, items):
    gpu_present = _gpu_physically_present()
    cuda_available = torch.cuda.is_available()

    for item in items:
        if "gpu" not in item.keywords:
            continue
        if not gpu_present:
            item.add_marker(pytest.mark.skip(reason="No NVIDIA GPU detected"))
        elif not cuda_available:
            item.add_marker(pytest.mark.xfail(
                reason="GPU detected but torch cannot use it — "
                       "reinstall torch with the correct CUDA version: "
                       "pip install torch --extra-index-url "
                       "https://download.pytorch.org/whl/cu<VERSION>",
                strict=True,
            ))
