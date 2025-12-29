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
python scripts/eval_pick.py --ckpt checkpoints/latest.json --config configs/train_pick.yaml
python scripts/eval_pick.py --ckpt checkpoints/latest.json --config configs/train_pick.yaml --render
python scripts/eval_pick.py --ckpt checkpoints/latest.json --config configs/train_pick.yaml --record --record-dir videos
python scripts/eval_pick.py --ckpt checkpoints/latest.json --config configs/train_pick.yaml --inline
```

注意：
- `scripts/train_pick.py` 目前是隨機策略的最小流程，之後需替換成 TD-MPC 整合。
- 在 Colab/無頭環境建議設定 `MUJOCO_GL=egl` 以使用 GPU 渲染。
- 若出現 Mesa shader cache 權限錯誤，請設定 `XDG_CACHE_HOME` 到可寫目錄。
- 若安裝時遇到 pip constraint 衝突，請用 `PIP_CONSTRAINT=` 清空限制。

## Repo 結構

```
configs/
  env/lowcostrobot_pick.yaml
  algo/tdmpc.yaml
  train_pick.yaml
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
  utils_seed.py
  utils_logging.py
logs/
checkpoints/
third_party/
```

## 常見問題
- MuJoCo import 失敗：確認已安裝 `mujoco`，Colab 請設定 `MUJOCO_GL=egl`。
- Env 找不到：確認 `third_party/Sim-LeRobotHackathon` 存在且包含指定的 env id。
