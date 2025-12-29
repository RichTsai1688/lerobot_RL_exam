# spec.md

# RL Pick Baseline（Low-cost robot arm in MuJoCo）Spec

## 1. 目標與範圍
建立一個可重現、可擴充的「機械手臂取物（Pick / Pick&Place）」RL 訓練專案：
- MVP：能在模擬中學會 **抓起方塊（Pick）**，並輸出成功率曲線與 checkpoint。
- Baseline+：加入 domain randomization、評估 protocol、穩定化訓練設定。
- 進階：支援示範資料（teleop/BC）+ RL fine-tune、影像觀測（可選）。

## 2. 上游依賴與整合方式
上游環境：
- 使用 third_party/Sim-LeRobotHackathon（fork 或 submodule），其內含 gym-lowcostrobot + MuJoCo manipulation 任務。
- 訓練端以 LeRobot / TD-MPC 為主要 baseline（MVP 先跑通 pipeline）。

整合策略：
- third_party 放上游專案
- envs/lowcostrobot_pick/ 實作 wrapper / reward / randomization
- scripts/ 提供 train/eval/teleop 入口
- configs/ 統一管理環境與訓練參數

## 3. 成功標準（Acceptance Criteria）
### MVP
- 可以一鍵執行訓練：`python scripts/train_pick.py --config configs/train_pick.yaml`
- 訓練過程有 log（tensorboard 或 wandb 二擇一），至少包含：
  - episode_return
  - success_rate（以 eval 回合統計）
  - episode_length
- 有 eval 腳本可載入 checkpoint：`python scripts/eval_pick.py --ckpt <path>`
- 能在固定 seed 的評估設定中達到：
  - Pick 任務 success_rate >= 0.30（初版門檻，可調整但需可見上升趨勢）

### Baseline+
- 支援 domain randomization（摩擦/質量/初始位置/目標位置）
- eval protocol 固定（N 回合、固定 seed 列表、輸出 summary.json）

## 4. 任務定義（Pick / Pick&Place）
### Observation（MVP：state-based）
- robot joint positions q
- robot joint velocities dq
- gripper state（開合/指令/接觸狀態若可得）
- object pose（位置 xyz、姿態 quaternion）
- target pose（若是 pick&place）
> 影像（rgb/rgbd）屬於進階選項，MVP 不納入。

### Action
- 連續控制（arm joints / end-effector delta / gripper open-close）
- action smoothing / clamp（避免抖動與爆衝）

### Reward shaping（分段式，權重可配置）
- r_reach：末端到物體距離（負距離）
- r_grasp：抓取事件 bonus（接觸且夾爪閉合到位）
- r_lift：物體高度超過阈值 bonus（z > z_lift）
- r_place：放置成功 bonus（進入目標區且穩定 K steps）
- penalties：
  - action magnitude penalty
  - joint velocity penalty
  - time penalty

### Termination
- success（達成 lift 或 place）
- timeout（最大步數）
- unstable / out-of-bounds（可選）

## 5. 評估與指標
- eval 每隔 M steps 執行一次
- 指標：
  - success_rate
  - mean_return
  - mean_episode_length
  - grasp_count（可選）
- 輸出：
  - logs/ 下的 run 目錄
  - checkpoints/
  - eval_summary.json（包含 seeds、success、returns）

## 6. Repo 結構（目標狀態）
```
rl-pick-baseline/
README.md
pyproject.toml 或 requirements.txt
third_party/
Sim-LeRobotHackathon/   # fork/submodule
envs/
lowcostrobot_pick/
**init**.py
wrappers.py
rewards.py
domain_randomization.py
configs/
env/lowcostrobot_pick.yaml
algo/tdmpc.yaml
train_pick.yaml
scripts/
train_pick.py
eval_pick.py
record_teleop.py   # optional
logs/
checkpoints/

```

## 7. 非功能需求（NFR）
- 可重現：固定 seed、保存 config、保存 git commit hash（若可）
- 可移植：Linux/WSL/mac（以 MuJoCo 可用為前提）
- 可讀性：所有 reward/termination 參數集中在 configs
- 可擴充：未來可切換到影像 encoder、加入 BC pretrain

## 8. 風險與對策
- reward 不適配導致學不動：先用 dense shaping + 小目標（reach->grasp->lift）
- 動作空間不穩：加入 clamp + smoothing
- sim 參數敏感：加入 domain randomization + eval 固定 seeds

## 9. 里程碑
- M0：能跑通上游 env demo（PushCube）
- M1：Pick MVP（成功率曲線上升）
- M2：Baseline+（randomization + 固定 eval）
- M3：可選（teleop/BC + RL、影像觀測）
