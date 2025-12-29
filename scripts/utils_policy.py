from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from gymnasium.spaces import Box


@dataclass
class GpuPolicy:
    model: torch.nn.Module
    action_low: torch.Tensor
    action_high: torch.Tensor
    device: torch.device
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]


def _assert_box_space(action_space: Box) -> None:
    if not isinstance(action_space, Box):
        raise TypeError("GPU policy requires a Box action space.")


def _flatten_obs(obs: np.ndarray | dict) -> np.ndarray:
    if isinstance(obs, dict):
        # Stable ordering so the policy input is deterministic.
        parts = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, dict):
                raise TypeError("Nested dict observations are not supported.")
            parts.append(np.asarray(value).reshape(-1))
        if not parts:
            raise TypeError("Empty dict observation is not supported.")
        return np.concatenate(parts, axis=0)
    return np.asarray(obs).reshape(-1)


def create_gpu_policy(obs_sample: np.ndarray | dict, action_space: Box) -> GpuPolicy:
    _assert_box_space(action_space)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    obs_flat = _flatten_obs(obs_sample)
    if obs_flat.dtype == np.object_:
        raise TypeError("GPU policy expects a numeric observation array.")

    obs_dim = int(np.prod(obs_flat.shape))
    act_dim = int(np.prod(action_space.shape))
    device = torch.device("cuda")
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, act_dim),
        torch.nn.Tanh(),
    ).to(device)

    action_low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32).reshape(-1)
    action_high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32).reshape(-1)
    return GpuPolicy(
        model=model,
        action_low=action_low,
        action_high=action_high,
        device=device,
        obs_shape=obs_flat.shape,
        action_shape=action_space.shape,
    )


def policy_action(policy: GpuPolicy, obs: np.ndarray | dict) -> np.ndarray:
    flat_obs = _flatten_obs(obs)
    obs_tensor = torch.as_tensor(flat_obs, device=policy.device, dtype=torch.float32)

    with torch.no_grad():
        raw = policy.model(obs_tensor)
        # Map tanh output (-1,1) to action bounds.
        scaled = (raw + 1.0) / 2.0
        action = policy.action_low + scaled * (policy.action_high - policy.action_low)

    return action.reshape(policy.action_shape).cpu().numpy()
