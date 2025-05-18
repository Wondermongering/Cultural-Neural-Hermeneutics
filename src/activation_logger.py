from collections import defaultdict
from typing import Dict, List

import torch


class ActivationLogger:
    """Collect activations from hooks."""

    def __init__(self):
        self.data: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def hook_fn(self, name: str):
        def _hook(module, inp, out):
            self.data[name].append(out.detach().cpu())
        return _hook
