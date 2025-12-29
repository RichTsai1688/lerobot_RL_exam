import argparse

from envs.lowcostrobot_pick import create_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PushCube-v0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    try:
        env = create_env(args.env_id)
    except Exception as exc:
        print("Failed to create env. Ensure third_party/Sim-LeRobotHackathon is cloned and env id exists.")
        print(f"Error: {exc}")
        return 1

    for ep in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(args.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        print(f"episode {ep} ok")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
