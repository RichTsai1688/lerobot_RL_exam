import argparse
import json
import os
import sys
import time
from typing import Any, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import yaml

from envs.lowcostrobot_pick import create_env
from scripts.utils_cuda import print_cuda_diagnostics
from scripts.utils_policy import create_gpu_policy, policy_action
from scripts.utils_logging import create_run_dir, save_config_snapshot, write_run_metadata
from scripts.utils_seed import set_seed


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    env_cfg = _load_yaml(cfg["env_config"])

    set_seed(cfg.get("seed", 42))
    print_cuda_diagnostics(prefix="train_cuda")

    env = create_env(env_cfg["env_id"], seed=env_cfg.get("seed"))
    assert isinstance(env, gym.Env)

    run_dir = create_run_dir(cfg.get("run_dir", "logs"))
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    save_config_snapshot({"train": cfg, "env": env_cfg}, run_dir)
    write_run_metadata(run_dir)

    total_steps = int(cfg.get("steps", 2000))
    eval_interval = int(cfg.get("eval_interval", 500))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 500))

    try:
        obs, _ = env.reset()
        policy = create_gpu_policy(obs, env.action_space)
        ep_return = 0.0
        ep_start = time.perf_counter()
        episode = 0
        ep_steps = 0
        success = None
        step = 0

        while step < total_steps:
            action = policy_action(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
            ep_steps += 1
            if success is None:
                success = info.get("success")

            if terminated or truncated:
                elapsed = time.perf_counter() - ep_start
                success_text = "n/a" if success is None else str(bool(success))
                print(
                    f"episode {episode} return={ep_return:.4f} "
                    f"steps={ep_steps} time_s={elapsed:.3f} success={success_text}"
                )
                obs, _ = env.reset()
                ep_return = 0.0
                ep_start = time.perf_counter()
                ep_steps = 0
                success = None
                episode += 1

            if step % eval_interval == 0:
                print(f"step={step} (placeholder eval)")

            if step % checkpoint_interval == 0:
                ckpt_path = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "latest.json")
                _save_checkpoint(ckpt_path, {"step": step, "note": "placeholder checkpoint"})
                print(f"checkpoint saved: {ckpt_path}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
