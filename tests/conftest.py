import pytest
import torch


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
