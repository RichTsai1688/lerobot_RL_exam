# RL Pick Baseline

這是 MuJoCo 上可重現的取物（Pick）RL baseline 專案最小骨架，對齊 `spec.md` 與 `plan.md`。

## 快速開始（本機）

```bash
# 1) 安裝依賴
python -m venv .venv
source .venv/bin/activate
PIP_CONSTRAINT= pip install -r requirements.txt

# 2) 取得上游環境
mkdir -p third_party
cd third_party
git clone https://github.com/perezjln/gym-lowcostrobot.git
cd ..

# 3) Smoke test (無頭環境請用 EGL)
export MUJOCO_GL=egl
export XDG_CACHE_HOME="$(pwd)/.cache"
python scripts/smoke_test_env.py --env-id PushCube-v0
```

## 快速開始（Colab，T4）

```bash
# 在 Colab cell 中
!bash colab_setup.sh

# 然後執行
!python scripts/smoke_test_env.py --env-id PushCube-v0
```

## 訓練 / 評估（目前為 placeholder runner）

```bash
python scripts/train_pick.py --config configs/train_pick.yaml
python scripts/eval_pick.py --ckpt checkpoints/latest.zip --config configs/train_pick.yaml
python scripts/eval_pick.py --ckpt checkpoints/latest.zip --config configs/train_pick.yaml --render
python scripts/eval_pick.py --ckpt checkpoints/latest.zip --config configs/train_pick.yaml --record --record-dir videos
python scripts/eval_pick.py --ckpt checkpoints/latest.zip --config configs/train_pick.yaml --inline

# TD-MPC (state-only)
python scripts/train_tdmpc.py --config configs/train_tdmpc.yaml
python scripts/eval_tdmpc.py --ckpt checkpoints/tdmpc_latest.pt --config configs/train_tdmpc.yaml
python scripts/eval_tdmpc.py --ckpt checkpoints/tdmpc_latest.pt --config configs/train_tdmpc.yaml --render
python scripts/eval_tdmpc.py --ckpt checkpoints/tdmpc_latest.pt --config configs/train_tdmpc.yaml --record --record-dir videos
```

注意：
- `scripts/train_pick.py` 目前使用 Stable-Baselines3 的 SAC 作為 baseline。
- `scripts/train_tdmpc.py` 預設用 state-only observation，若要影像觀測請在 `configs/algo/tdmpc.yaml` 設定 `image_keys`。
- 影像觀測會顯著增加算力需求與訓練時間，建議先用 state-only 確認流程。
- 在 Colab/無頭環境建議設定 `MUJOCO_GL=egl` 以使用 GPU 渲染。
- 若出現 Mesa shader cache 權限錯誤，請設定 `XDG_CACHE_HOME` 到可寫目錄。
- 若安裝時遇到 pip constraint 衝突，請用 `PIP_CONSTRAINT=` 清空限制。

## 設定檔說明

可選演算法與設定（目前僅 placeholder 範例）：

| 類型 | 名稱 | 設定檔 | 說明 |
| --- | --- | --- | --- |
| algo | sac | configs/algo/sac.yaml | Stable-Baselines3 SAC 設定 |
| algo | tdmpc | configs/algo/tdmpc.yaml | TD-MPC (state-only) 設定 |
| env | lowcostrobot_pick | configs/env/lowcostrobot_pick.yaml | Low-cost robot pick 環境設定 |

`configs/train_pick.yaml`
- `seed`: 隨機種子，控制可重現性
- `steps`: 訓練總步數（目前是隨機 policy 的步數）
- `eval_interval`: 每隔多少步印一次 eval 訊息（placeholder）
- `checkpoint_interval`: 每隔多少步寫一次 checkpoint（目前為 50）
- `run_dir`: 訓練記錄輸出資料夾
- `checkpoint_dir`: checkpoint 輸出資料夾
- `env_config`: 環境設定檔路徑
- `algo_config`: 演算法設定檔路徑（目前是 placeholder）

`configs/env/lowcostrobot_pick.yaml`
- `env_id`: Gymnasium 環境 ID
- `max_episode_steps`: 每回合最大步數（需要 wrapper 套用）
- `seed`: 環境隨機種子
- `reward.*`: reward 權重與懲罰（placeholder）
- `termination.*`: 終止條件（placeholder）
- `randomization.*`: domain randomization 設定

`configs/algo/sac.yaml`
- `name`: 演算法名稱
- `policy`: SB3 policy 類型
- `device`: 計算裝置（auto/cpu/cuda）
- `learning_rate`, `buffer_size`, `batch_size`, `gamma`, `tau`: SAC 主要超參數
- `train_freq`, `gradient_steps`, `learning_starts`: 訓練頻率與 warmup
- `ent_coef`, `target_entropy`: 自動 entropy 設定

`configs/algo/tdmpc.yaml`
- `state_keys`: TD-MPC 使用的 state observation keys
- `image_keys`: 啟用影像觀測時要使用的 image keys（例如 `image_front`, `image_top`）
- `latent_dim`, `hidden_dim`: encoder / model 隱層
- `state_embed_dim`, `image_embed_dim`: state / image encoder embedding 維度
- `batch_size`, `lr`, `gamma`, `tau`: 主要超參數
- `replay_size`, `start_steps`, `train_freq`, `gradient_steps`: replay 與更新頻率
- `policy_bc_weight`: policy 行為複製權重（用 planner 動作監督）
- `cem_*`, `horizon`, `action_noise`: CEM 規劃設定

## Repo 結構

```
configs/
  env/lowcostrobot_pick.yaml
  algo/tdmpc.yaml
  algo/sac.yaml
  train_pick.yaml
  train_tdmpc.yaml
envs/
  lowcostrobot_pick/
    __init__.py
    wrappers.py
    rewards.py
    domain_randomization.py
scripts/
  smoke_test_env.py
  train_pick.py
  eval_pick.py
  train_tdmpc.py
  eval_tdmpc.py
  tdmpc_agent.py
  utils_seed.py
  utils_logging.py
logs/
checkpoints/
third_party/
```

## 常見問題
- MuJoCo import 失敗：確認已安裝 `mujoco`，Colab 請設定 `MUJOCO_GL=egl`。
- Env 找不到：確認 `third_party/Sim-LeRobotHackathon` 存在且包含指定的 env id。
