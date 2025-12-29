import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import TimeLimit

from envs.lowcostrobot_pick import create_env
from scripts.tdmpc_agent import ReplayBuffer, StateObsWrapper, TDMPC, make_cem_config, pack_obs
from scripts.utils_cuda import print_cuda_diagnostics
from scripts.utils_logging import create_run_dir, save_config_snapshot, write_run_metadata
from scripts.utils_seed import set_seed


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _make_env(env_cfg: Dict[str, Any], state_keys: List[str], use_image: bool) -> gym.Env:
    env = create_env(env_cfg["env_id"], seed=env_cfg.get("seed"))
    if state_keys and not use_image:
        env = StateObsWrapper(env, state_keys)
    max_steps = env_cfg.get("max_episode_steps")
    if max_steps:
        env = TimeLimit(env, max_episode_steps=int(max_steps))
    return env


def _evaluate(agent: TDMPC, env: gym.Env, episodes: int) -> Tuple[float, float]:
    returns = []
    successes = []
    for _ in range(episodes):
        obs, _ = env.reset()
        agent.reset_planner()
        done = False
        ep_return = 0.0
        success = None
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
            if success is None:
                success = info.get("success")
        returns.append(ep_return)
        if success is not None:
            successes.append(float(bool(success)))
    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_success = float(np.mean(successes)) if successes else 0.0
    return mean_return, mean_success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    env_cfg = _load_yaml(cfg["env_config"])
    algo_cfg = _load_yaml(cfg["algo_config"])

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    print_cuda_diagnostics(prefix="train_cuda")

    state_keys = algo_cfg.get("state_keys", [])
    image_keys = algo_cfg.get("image_keys", [])
    use_image = bool(image_keys)
    env = _make_env(env_cfg, state_keys, use_image)
    eval_env = _make_env(env_cfg, state_keys, use_image)
    if not use_image and isinstance(env.observation_space, gym.spaces.Dict) and not state_keys:
        raise ValueError("state_keys must be set when using dict observations without images.")

    if use_image:
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Image observations require dict observation space.")
        image_space = env.observation_space.spaces[image_keys[0]]
        if not isinstance(image_space, gym.spaces.Box):
            raise TypeError("Image observations must be Box spaces.")
        image_shape = tuple(image_space.shape)
        for key in image_keys[1:]:
            space = env.observation_space.spaces[key]
            if not isinstance(space, gym.spaces.Box) or tuple(space.shape) != image_shape:
                raise ValueError("All image_keys must have the same Box shape.")
    else:
        image_shape = None

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        state_dim = int(np.prod(obs_space.shape))
    else:
        state_dim = 0
        for key in state_keys:
            space = obs_space.spaces[key]
            state_dim += int(np.prod(space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    run_dir = create_run_dir(cfg.get("run_dir", "logs"))
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    save_config_snapshot({"train": cfg, "env": env_cfg, "algo": algo_cfg}, run_dir)
    write_run_metadata(run_dir)

    total_steps = int(cfg.get("steps", 20000))
    eval_interval = int(cfg.get("eval_interval", 2000))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 2000))

    device_name = algo_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    cem = make_cem_config(algo_cfg)

    agent = TDMPC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        latent_dim=int(algo_cfg.get("latent_dim", 64)),
        hidden_dim=int(algo_cfg.get("hidden_dim", 256)),
        lr=float(algo_cfg.get("lr", 3e-4)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        tau=float(algo_cfg.get("tau", 0.01)),
        cem=cem,
        device=device,
        state_embed_dim=int(algo_cfg.get("state_embed_dim", 64)),
        image_embed_dim=int(algo_cfg.get("image_embed_dim", 64)),
        image_shape=image_shape,
        num_images=len(image_keys),
        state_keys=state_keys if use_image else [],
        image_keys=image_keys,
        policy_bc_weight=float(algo_cfg.get("policy_bc_weight", 1.0)),
    )

    buffer = ReplayBuffer(
        capacity=int(algo_cfg.get("replay_size", 100000)),
        state_dim=state_dim,
        action_dim=action_dim,
        image_shape=image_shape,
        num_images=len(image_keys),
    )

    start_steps = int(algo_cfg.get("start_steps", 1000))
    batch_size = int(algo_cfg.get("batch_size", 256))
    train_freq = int(algo_cfg.get("train_freq", 1))
    gradient_steps = int(algo_cfg.get("gradient_steps", 1))

    obs, _ = env.reset()
    agent.reset_planner()
    ep_return = 0.0
    ep_start = time.perf_counter()
    episode = 0
    step = 0

    try:
        while step < total_steps:
            if step < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, eval_mode=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state, images = pack_obs(obs, state_keys if use_image else [], image_keys)
            next_state, next_images = pack_obs(next_obs, state_keys if use_image else [], image_keys)
            buffer.add(state, action, float(reward), next_state, done, images=images, next_images=next_images)
            obs = next_obs
            ep_return += float(reward)
            step += 1

            if done:
                elapsed = time.perf_counter() - ep_start
                success = info.get("success")
                success_text = "n/a" if success is None else str(bool(success))
                print(
                    f"episode {episode} return={ep_return:.4f} time_s={elapsed:.3f} "
                    f"success={success_text}"
                )
                obs, _ = env.reset()
                agent.reset_planner()
                ep_return = 0.0
                ep_start = time.perf_counter()
                episode += 1

            if len(buffer) >= batch_size and step % train_freq == 0 and step >= start_steps:
                for _ in range(gradient_steps):
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                if step % eval_interval == 0:
                    mean_return, mean_success = _evaluate(agent, eval_env, episodes=3)
                    print(f"eval step={step} return={mean_return:.3f} success={mean_success:.3f}")

            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                ckpt_root = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "tdmpc_latest.pt")
                agent.save(ckpt_root)
                _save_checkpoint(
                    f"{ckpt_root}.json",
                    {"step": step, "episode": episode, "metrics": metrics if "metrics" in locals() else {}},
                )
                print(f"checkpoint saved: {ckpt_root}")

    finally:
        env.close()
        eval_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
