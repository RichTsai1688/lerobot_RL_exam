```yaml
# task.yaml
# RL Pick Baseline - Task Breakdown (for Codex)

project:
  name: rl-pick-baseline
  goal: Build a reproducible RL training baseline for robotic pick / pick&place in MuJoCo using Sim-LeRobotHackathon + TD-MPC.
  language: python
  priority: mvp_first

assumptions:
  - Upstream repo will be placed at third_party/Sim-LeRobotHackathon (git submodule or fork).
  - Training algo uses TD-MPC via LeRobot-style runner OR a lightweight local runner (choose the minimal integration that works).
  - MVP uses state observations, not images.

tasks:
  - id: T0_repo_scaffold
    title: Scaffold repository structure
    outputs:
      - README.md
      - configs/{env,algo}/
      - envs/lowcostrobot_pick/
      - scripts/
      - .gitignore
    acceptance:
      - Tree matches spec.md structure
      - Minimal README with quickstart commands

  - id: T1_upstream_integration
    title: Integrate Sim-LeRobotHackathon as submodule/folder
    steps:
      - Add third_party/Sim-LeRobotHackathon
      - Provide a small helper to import upstream gym envs
    acceptance:
      - `python -c "import third_party...; print('ok')"` (or equivalent import path)
      - A simple env can be created (e.g., PushCube) from our scripts

  - id: T2_env_wrapper_pick
    title: Implement Pick environment wrapper (obs/action/reward/termination)
    files:
      - envs/lowcostrobot_pick/wrappers.py
      - envs/lowcostrobot_pick/rewards.py
      - envs/lowcostrobot_pick/__init__.py
    requirements:
      - State obs vector assembly function
      - Action clamp + smoothing wrapper
      - Reward shaping functions with configurable weights
      - Termination logic (success/timeout/out-of-bounds optional)
    acceptance:
      - `python scripts/smoke_test_env.py` runs 10 episodes random policy without crash
      - Reward components can be toggled/weighted via config

  - id: T3_domain_randomization
    title: Add domain randomization utilities
    files:
      - envs/lowcostrobot_pick/domain_randomization.py
      - configs/env/lowcostrobot_pick.yaml
    requirements:
      - Randomize object initial pose within bounds
      - Randomize friction/mass (if supported by MuJoCo model handles)
      - Enable/disable via config
    acceptance:
      - When enabled, reset produces varied initial states across episodes
      - When disabled, deterministic resets with fixed seed

  - id: T4_training_entrypoint
    title: Training entrypoint script (TD-MPC)
    files:
      - scripts/train_pick.py
      - configs/train_pick.yaml
      - configs/algo/tdmpc.yaml
    requirements:
      - Load yaml config
      - Create env (with wrappers)
      - Launch training loop (via LeRobot runner or minimal TD-MPC integration)
      - Save checkpoints and training logs
    acceptance:
      - Command runs end-to-end and saves at least one checkpoint
      - Logs include return + success_rate (eval)

  - id: T5_eval_entrypoint
    title: Evaluation script
    files:
      - scripts/eval_pick.py
    requirements:
      - Load checkpoint
      - Run N eval episodes with fixed seeds
      - Output eval_summary.json with metrics
    acceptance:
      - Produces eval_summary.json
      - Prints summary to stdout

  - id: T6_logging_and_repro
    title: Logging + reproducibility helpers
    files:
      - scripts/utils_logging.py (optional)
      - scripts/utils_seed.py (optional)
    requirements:
      - Save full config copy to run directory
      - Record git commit hash if available
      - Set seeds for python/numpy/torch/env
    acceptance:
      - Re-running with same seed yields consistent eval results (within small tolerance)

  - id: T7_docs
    title: Documentation & quickstart
    outputs:
      - README.md updated with:
        - install steps
        - how to add upstream submodule
        - train/eval commands
        - troubleshooting (MuJoCo, GPU, dependencies)
    acceptance:
      - New user can follow README to run smoke test + train + eval

optional_tasks:
  - id: O1_teleop_data
    title: Teleop recording for demonstrations
    files:
      - scripts/record_teleop.py
    acceptance:
      - Saves trajectories to a dataset folder in a simple format (npz/jsonl)

  - id: O2_bc_plus_rl
    title: Behavior cloning pretrain then RL finetune
    acceptance:
      - Can train from demonstrations and then continue RL

  - id: O3_image_obs
    title: RGB/RGBD observation pipeline
    acceptance:
      - Adds an encoder and can train (even if slower)

order:
  - T0_repo_scaffold
  - T1_upstream_integration
  - T2_env_wrapper_pick
  - T4_training_entrypoint
  - T5_eval_entrypoint
  - T6_logging_and_repro
  - T3_domain_randomization
  - T7_docs
  - optional_tasks
```
