import argparse
import base64
import json
import os
from typing import Any, Dict, Optional

import yaml
from gymnasium.wrappers import RecordVideo

from envs.lowcostrobot_pick import create_env


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _latest_mp4(video_dir: str) -> Optional[str]:
    if not os.path.isdir(video_dir):
        return None
    mp4s = [os.path.join(video_dir, p) for p in os.listdir(video_dir) if p.endswith(".mp4")]
    if not mp4s:
        return None
    return max(mp4s, key=os.path.getmtime)


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

    try:
        with open(args.ckpt, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}")
        return 1

    record_enabled = args.record or args.inline
    render_mode = None
    if args.render:
        render_mode = "human"
    elif record_enabled:
        render_mode = "rgb_array"

    env = create_env(env_cfg["env_id"], seed=env_cfg.get("seed"), options={"render_mode": render_mode} if render_mode else None)
    if record_enabled:
        env = RecordVideo(env, video_folder=args.record_dir, episode_trigger=lambda _: True)

    returns = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            if args.render:
                env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)

    env.close()

    if args.inline:
        latest = _latest_mp4(args.record_dir)
        if latest:
            _maybe_inline_video(latest)

    summary = {
        "episodes": args.episodes,
        "mean_return": sum(returns) / len(returns),
        "checkpoint": ckpt,
    }
    with open("eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("eval_summary.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
