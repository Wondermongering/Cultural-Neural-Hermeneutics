import pytest

from src.activation_logger import ActivationLogger

pytest.importorskip("torch")
import torch
import torch.nn as nn


def test_logger_collects():
    model = nn.Linear(2, 2)
    logger = ActivationLogger()
    handle = model.register_forward_hook(logger.hook_fn("linear"))
    inp = torch.tensor([[1.0, 2.0]])
    _ = model(inp)
    handle.remove()
    assert "linear" in logger.data
    assert len(logger.data["linear"]) == 1
