import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def maybe_seed_env(env, seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        env.reset(seed=seed)
    except Exception:
        pass
