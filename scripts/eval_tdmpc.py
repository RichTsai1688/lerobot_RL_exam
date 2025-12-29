import argparse
import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import RecordVideo, TimeLimit

from envs.lowcostrobot_pick import create_env
from scripts.tdmpc_agent import StateObsWrapper, TDMPC, make_cem_config
from scripts.utils_cuda import print_cuda_diagnostics


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _latest_mp4(video_dir: str) -> Optional[str]:
    if not os.path.isdir(video_dir):
        return None
    mp4s = [os.path.join(video_dir, p) for p in os.listdir(video_dir) if p.endswith(".mp4")]
    if not mp4s:
        return None
    return max(mp4s, key=os.path.getmtime)


def _maybe_inline_video(video_path: str) -> None:
    try:
        from IPython.display import HTML, display
    except Exception:
        return

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = f"""
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    """
    display(HTML(html))


def _make_env(env_cfg: Dict[str, Any], state_keys: List[str], use_image: bool, render_mode: str | None = None) -> gym.Env:
    options = {"render_mode": render_mode} if render_mode else None
    env = create_env(env_cfg["env_id"], seed=env_cfg.get("seed"), options=options)
    if state_keys and not use_image:
        env = StateObsWrapper(env, state_keys)
    max_steps = env_cfg.get("max_episode_steps")
    if max_steps:
        env = TimeLimit(env, max_episode_steps=int(max_steps))
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    parser.add_argument("--record", action="store_true", help="Record evaluation videos to disk")
    parser.add_argument("--record-dir", default="videos", help="Directory for recorded videos")
    parser.add_argument("--inline", action="store_true", help="Inline display of the latest video (Colab/IPython)")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    env_cfg = _load_yaml(cfg["env_config"])
    algo_cfg = _load_yaml(cfg["algo_config"])

    print_cuda_diagnostics(prefix="eval_cuda")

    record_enabled = args.record or args.inline
    render_mode = None
    if args.render:
        render_mode = "human"
    elif record_enabled:
        render_mode = "rgb_array"

    state_keys = algo_cfg.get("state_keys", [])
    image_keys = algo_cfg.get("image_keys", [])
    use_image = bool(image_keys)
    env = _make_env(env_cfg, state_keys, use_image, render_mode=render_mode)
    if not use_image and isinstance(env.observation_space, gym.spaces.Dict) and not state_keys:
        raise ValueError("state_keys must be set when using dict observations without images.")
    if record_enabled:
        env = RecordVideo(env, video_folder=args.record_dir, episode_trigger=lambda _: True)

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
    device_name = algo_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
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
        cem=make_cem_config(algo_cfg),
        device=device,
        state_embed_dim=int(algo_cfg.get("state_embed_dim", 64)),
        image_embed_dim=int(algo_cfg.get("image_embed_dim", 64)),
        image_shape=image_shape,
        num_images=len(image_keys),
        state_keys=state_keys if use_image else [],
        image_keys=image_keys,
        policy_bc_weight=float(algo_cfg.get("policy_bc_weight", 1.0)),
    )

    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return 1
    agent.load(args.ckpt)

    returns = []
    successes = []
    try:
        for _ in range(args.episodes):
            obs, _ = env.reset()
            agent.reset_planner()
            done = False
            ep_return = 0.0
            success = None
            while not done:
                if args.render:
                    env.render()
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
                if success is None:
                    success = info.get("success")
            returns.append(ep_return)
            if success is not None:
                successes.append(float(bool(success)))
    finally:
        env.close()

    if args.inline:
        latest = _latest_mp4(args.record_dir)
        if latest:
            _maybe_inline_video(latest)

    summary = {
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_success": float(np.mean(successes)) if successes else 0.0,
        "checkpoint": args.ckpt,
    }
    with open("eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("eval_summary.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
