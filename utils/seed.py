from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，方便复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
