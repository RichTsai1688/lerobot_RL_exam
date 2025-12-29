from dataclasses import dataclass
from typing import Dict


@dataclass
class RandomizationConfig:
    enabled: bool = False
    object_pos_range: tuple[float, float, float] = (0.02, 0.02, 0.0)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    friction_scale_range: tuple[float, float] = (0.8, 1.2)


def apply_randomization(env, cfg: RandomizationConfig) -> Dict[str, float]:
    if not cfg.enabled:
        return {}
    # Placeholder: extend with MuJoCo model handles when upstream env is available.
    return {
        "mass_scale": 1.0,
        "friction_scale": 1.0,
    }
