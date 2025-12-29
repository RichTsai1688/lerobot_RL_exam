# RL Pick Baseline Plan

## Phase 0 — Bring-up（先把管線跑通）
**目標**：確保上游環境可跑、可 reset/step、可渲染（如果需要）。
1. 將 Sim-LeRobotHackathon 放入 third_party（submodule 或 fork）
2. 寫 smoke test：建立 env（先 PushCube），跑 1~2 個 episode
3. 確認依賴：mujoco、gym 相關、torch（若 TD-MPC 需要）

Deliverable：
- scripts/smoke_test_env.py（可選，但建議加）
- README Quickstart（先能跑 smoke test）

## Phase 1 — Pick MVP（state-based + dense shaping）
**目標**：做出可學會「抓起」的最小版本。
1. Wrapper：將 obs 組成 state vector（q,dq,object pose, gripper state）
2. Reward：reach → grasp → lift 的分段 shaping
3. Termination：lift success or timeout
4. TD-MPC 訓練入口：train_pick.py + configs
5. eval：固定 seeds 的 success_rate

驗收：
- 訓練曲線可見 success_rate 上升
- eval_pick.py 能輸出 eval_summary.json

## Phase 2 — Baseline+（穩定化與泛化）
**目標**：提升穩定性、可重現性、泛化能力。
1. domain randomization（初始位姿、摩擦、質量）
2. action smoothing/clamp 的超參數調整
3. eval protocol 標準化：
   - 固定 seed 列表
   - 每次 eval N 回合
   - 輸出 summary + best checkpoint

驗收：
- DR 開啟後，仍能維持一定 success_rate
- seed 重跑結果可重現（允許小波動）

## Phase 3 — Optional（示範 + 影像）
**目標**：更接近真實取物應用。
1. teleop 收示範資料（10~30 分鐘）
2. BC pretrain → RL finetune（縮短探索）
3. 影像觀測（rgb/rgbd）加入 encoder

驗收：
- BC+RL 的 learning curve 明顯優於純 RL（至少在 sample efficiency 上）

## Engineering Conventions
- configs 全部 YAML 化，run 時保存 config snapshot
- logs 目錄以 run_id 命名，含 tensorboard/wandb、eval_summary
- checkpoints 定期保存 + best 保存
- scripts 盡量無硬編碼路徑：都由 config 控制

## 建議的預設超參數（MVP）
- episode_length：200~400 steps（依上游 env）
- eval_interval：每 5k~10k steps
- reward weights（初始建議）：
  - w_reach=1.0
  - w_grasp=5.0
  - w_lift=10.0
  - penalties: action=0.01, vel=0.001, time=0.001
- lift threshold z_lift：依方塊尺寸與桌面高度（config 化）

## 交付清單
- spec.md（本文件對應的規格）
- task.yaml（可給 Codex 執行的任務拆解）
- plan.md（分階段計畫與驗收）
