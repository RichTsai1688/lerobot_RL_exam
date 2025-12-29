import os
import sys
from typing import Any, Dict

import gymnasium as gym


def _ensure_upstream_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    upstream = os.path.join(repo_root, "third_party", "gym-lowcostrobot")
    if os.path.isdir(upstream) and upstream not in sys.path:
        sys.path.insert(0, upstream)


def create_env(env_id: str, seed: int | None = None, options: Dict[str, Any] | None = None) -> gym.Env:
    _ensure_upstream_on_path()
    # Import to ensure gym-lowcostrobot registers its envs with Gymnasium.
    try:
        import gym_lowcostrobot  # noqa: F401
    except ImportError:
        pass
    env = gym.make(env_id, **(options or {}))
    if seed is not None:
        env.reset(seed=seed)
    return env
