import argparse
import json
import os
from typing import Any, Dict

import gymnasium as gym
import yaml

from envs.lowcostrobot_pick import create_env
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

    env = create_env(env_cfg["env_id"], seed=env_cfg.get("seed"))
    assert isinstance(env, gym.Env)

    run_dir = create_run_dir(cfg.get("run_dir", "logs"))
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    save_config_snapshot({"train": cfg, "env": env_cfg}, run_dir)
    write_run_metadata(run_dir)

    total_steps = int(cfg.get("steps", 2000))
    eval_interval = int(cfg.get("eval_interval", 500))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 500))

    obs, _ = env.reset()
    ep_return = 0.0
    step = 0

    while step < total_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        step += 1

        if terminated or truncated:
            obs, _ = env.reset()
            ep_return = 0.0

        if step % eval_interval == 0:
            print(f"step={step} (placeholder eval)")

        if step % checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "latest.json")
            _save_checkpoint(ckpt_path, {"step": step, "note": "placeholder checkpoint"})
            print(f"checkpoint saved: {ckpt_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
