import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class StateObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, keys: Iterable[str]) -> None:
        super().__init__(env)
        self._keys = list(keys)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("StateObsWrapper expects a dict observation space.")
        missing = [key for key in self._keys if key not in env.observation_space.spaces]
        if missing:
            raise KeyError(f"Missing observation keys: {missing}")

        lows = []
        highs = []
        for key in self._keys:
            space = env.observation_space.spaces[key]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"Observation key '{key}' must be a Box space.")
            lows.append(space.low.reshape(-1))
            highs.append(space.high.reshape(-1))
        low = np.concatenate(lows, axis=0).astype(np.float32)
        high = np.concatenate(highs, axis=0).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: dict) -> np.ndarray:
        parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in self._keys]
        return np.concatenate(parts, axis=0)


def pack_obs(
    obs: np.ndarray | Dict[str, np.ndarray],
    state_keys: List[str],
    image_keys: List[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(obs, dict):
        state_parts = []
        for key in state_keys:
            state_parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        state = np.concatenate(state_parts, axis=0) if state_parts else np.zeros((0,), dtype=np.float32)
        if image_keys:
            images = [np.asarray(obs[key], dtype=np.uint8) for key in image_keys]
            image_stack = np.stack(images, axis=0)
        else:
            image_stack = None
        return state, image_stack
    return np.asarray(obs, dtype=np.float32).reshape(-1), None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        image_shape: Optional[Tuple[int, int, int]] = None,
        num_images: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.images = None
        self.next_images = None
        if image_shape and num_images > 0:
            h, w, c = image_shape
            self.images = np.zeros((self.capacity, num_images, h, w, c), dtype=np.uint8)
            self.next_images = np.zeros((self.capacity, num_images, h, w, c), dtype=np.uint8)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        images: Optional[np.ndarray] = None,
        next_images: Optional[np.ndarray] = None,
    ) -> None:
        self.state[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_state[self.idx] = next_state
        self.dones[self.idx] = float(done)
        if self.images is not None and images is not None:
            self.images[self.idx] = images
        if self.next_images is not None and next_images is not None:
            self.next_images[self.idx] = next_images
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, Optional[np.ndarray]]:
        max_idx = self.capacity if self.full else self.idx
        idx = np.random.randint(0, max_idx, size=batch_size)
        batch = {
            "state": self.state[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_state[idx],
            "done": self.dones[idx],
            "images": None,
            "next_images": None,
        }
        if self.images is not None:
            batch["images"] = self.images[idx]
        if self.next_images is not None:
            batch["next_images"] = self.next_images[idx]
        return batch


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], embed_dim: int) -> None:
        super().__init__()
        h, w, c = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_dim = int(np.prod(conv_out.shape[1:]))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


class ObsEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_images: int,
        image_shape: Optional[Tuple[int, int, int]],
        hidden_dim: int,
        latent_dim: int,
        state_embed_dim: int,
        image_embed_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_images = num_images
        self.image_shape = image_shape
        self.state_mlp = MLP(state_dim, state_embed_dim, hidden_dim) if state_dim > 0 else None
        self.image_encoder = ImageEncoder(image_shape, image_embed_dim) if num_images > 0 and image_shape else None
        fusion_in = 0
        if state_dim > 0:
            fusion_in += state_embed_dim
        if num_images > 0:
            fusion_in += image_embed_dim * num_images
        self.fusion = MLP(fusion_in, latent_dim, hidden_dim)

    def forward(self, state: torch.Tensor, images: Optional[torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.state_mlp is not None:
            feats.append(self.state_mlp(state))
        if self.image_encoder is not None and images is not None:
            img = images.float() / 255.0
            img = img.permute(0, 1, 4, 2, 3)
            b, n, c, h, w = img.shape
            img = img.reshape(b * n, c, h, w)
            img_feat = self.image_encoder(img)
            img_feat = img_feat.reshape(b, n * img_feat.shape[-1])
            feats.append(img_feat)
        if not feats:
            raise RuntimeError("ObsEncoder received no inputs.")
        fused = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        return self.fusion(fused)


@dataclass
class CEMConfig:
    horizon: int
    population: int
    elite: int
    iterations: int
    init_std: float
    min_std: float
    action_noise: float


class TDMPC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        latent_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        cem: CEMConfig,
        device: torch.device,
        state_embed_dim: int = 64,
        image_embed_dim: int = 64,
        image_shape: Optional[Tuple[int, int, int]] = None,
        num_images: int = 0,
        state_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None,
        policy_bc_weight: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.cem = cem
        self.device = device
        self.policy_bc_weight = float(policy_bc_weight)
        self.state_keys = state_keys or []
        self.image_keys = image_keys or []
        self.num_images = num_images
        self.image_shape = image_shape

        self.encoder = ObsEncoder(
            state_dim=state_dim,
            num_images=num_images,
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            state_embed_dim=state_embed_dim,
            image_embed_dim=image_embed_dim,
        ).to(device)
        self.dynamics = MLP(latent_dim + action_dim, latent_dim, hidden_dim).to(device)
        self.reward = MLP(latent_dim + action_dim, 1, hidden_dim).to(device)
        self.value = MLP(latent_dim, 1, hidden_dim).to(device)
        self.policy = MLP(latent_dim, action_dim, hidden_dim).to(device)
        self.target_value = copy.deepcopy(self.value).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._prev_mean = None

    def parameters(self) -> List[nn.Parameter]:
        params = []
        for module in (self.encoder, self.dynamics, self.reward, self.value, self.policy):
            params.extend(module.parameters())
        return params

    def _encode(self, state: torch.Tensor, images: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(state, images)

    def _policy(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(z))

    def act(self, obs: np.ndarray | Dict[str, np.ndarray], eval_mode: bool = False) -> np.ndarray:
        state_np, images_np = pack_obs(obs, self.state_keys, self.image_keys)
        state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        images_tensor = None
        if images_np is not None:
            images_tensor = torch.as_tensor(images_np, dtype=torch.uint8, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z0 = self._encode(state_tensor, images_tensor)
            mean = self._init_mean(z0)
            action = self._cem_plan(z0, mean)
        if not eval_mode and self.cem.action_noise > 0:
            noise = torch.randn_like(action) * self.cem.action_noise
            action = action + noise
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.squeeze(0).cpu().numpy()

    def reset_planner(self) -> None:
        self._prev_mean = None

    def _init_mean(self, z0: torch.Tensor) -> torch.Tensor:
        if self._prev_mean is None:
            action = self._policy(z0)
            return action.repeat(self.cem.horizon, 1)
        prev = self._prev_mean
        shifted = torch.cat([prev[1:], prev[-1:]], dim=0)
        return shifted

    def _cem_plan(self, z0: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        mean = mean.clone().to(self.device)
        std = torch.ones_like(mean) * self.cem.init_std
        for _ in range(self.cem.iterations):
            noise = torch.randn((self.cem.population, self.cem.horizon, self.action_dim), device=self.device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            samples = torch.max(torch.min(samples, self.action_high), self.action_low)
            scores = self._score_sequences(z0, samples)
            elite_idx = torch.topk(scores, k=self.cem.elite, dim=0).indices
            elite = samples[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp(min=self.cem.min_std)
        self._prev_mean = mean.detach()
        return mean[0:1]

    def _score_sequences(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch = actions.shape[0]
        z = z0.repeat(batch, 1)
        total = torch.zeros((batch, 1), device=self.device)
        discount = 1.0
        for t in range(self.cem.horizon):
            a_t = actions[:, t, :]
            za = torch.cat([z, a_t], dim=-1)
            r = self.reward(za)
            total = total + discount * r
            z = self.dynamics(za)
            discount *= self.gamma
        total = total + discount * self.value(z)
        return total.squeeze(-1)

    def update(self, batch: Dict[str, Optional[np.ndarray]]) -> dict:
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        images = batch.get("images")
        next_images = batch.get("next_images")
        images_t = torch.as_tensor(images, dtype=torch.uint8, device=self.device) if images is not None else None
        next_images_t = torch.as_tensor(next_images, dtype=torch.uint8, device=self.device) if next_images is not None else None

        z = self._encode(state, images_t)
        with torch.no_grad():
            z_next = self._encode(next_state, next_images_t)
            target_v = self.target_value(z_next)
            td_target = rewards_t + self.gamma * (1.0 - dones_t) * target_v

        pred_z_next = self.dynamics(torch.cat([z, actions_t], dim=-1))
        pred_reward = self.reward(torch.cat([z, actions_t], dim=-1))
        pred_value = self.value(z)
        pred_action = self._policy(z)

        loss_dyn = torch.mean((pred_z_next - z_next) ** 2)
        loss_reward = torch.mean((pred_reward - rewards_t) ** 2)
        loss_value = torch.mean((pred_value - td_target) ** 2)
        loss_policy = torch.mean((pred_action - actions_t) ** 2)
        loss = loss_dyn + loss_reward + loss_value + self.policy_bc_weight * loss_policy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._soft_update(self.value, self.target_value, self.tau)

        return {
            "loss_total": float(loss.item()),
            "loss_dyn": float(loss_dyn.item()),
            "loss_reward": float(loss_reward.item()),
            "loss_value": float(loss_value.item()),
            "loss_policy": float(loss_policy.item()),
        }

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters()):
                tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    def save(self, path: str) -> None:
        payload = {
            "encoder": self.encoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "reward": self.reward.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
            "target_value": self.target_value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.cpu().numpy(),
            "action_high": self.action_high.cpu().numpy(),
            "num_images": self.num_images,
            "image_shape": self.image_shape,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(payload["encoder"])
        self.dynamics.load_state_dict(payload["dynamics"])
        self.reward.load_state_dict(payload["reward"])
        self.value.load_state_dict(payload["value"])
        self.policy.load_state_dict(payload["policy"])
        self.target_value.load_state_dict(payload["target_value"])
        self.optimizer.load_state_dict(payload["optimizer"])


def make_cem_config(cfg: dict) -> CEMConfig:
    return CEMConfig(
        horizon=int(cfg.get("horizon", 5)),
        population=int(cfg.get("cem_population", 256)),
        elite=int(cfg.get("cem_elite", 32)),
        iterations=int(cfg.get("cem_iters", 3)),
        init_std=float(cfg.get("cem_init_std", 0.5)),
        min_std=float(cfg.get("cem_min_std", 0.05)),
        action_noise=float(cfg.get("action_noise", 0.1)),
    )
