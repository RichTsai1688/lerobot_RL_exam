import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.lowcostrobot_pick import create_env
from scripts.utils_cuda import print_cuda_diagnostics
from scripts.utils_policy import create_gpu_policy, policy_action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PushCube-v0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    print_cuda_diagnostics(prefix="smoke_cuda")

    try:
        env = create_env(args.env_id)
    except Exception as exc:
        print("Failed to create env. Ensure third_party/gym-lowcostrobot is cloned and env id exists.")
        print(f"Error: {exc}")
        return 1

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            policy = create_gpu_policy(obs, env.action_space)
            for _ in range(args.max_steps):
                action = policy_action(policy, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            print(f"episode {ep} ok")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
