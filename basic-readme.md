# Basic README

這份文件提供本專案的基礎知識與模型/流程概念，適合第一次接觸時快速理解。

## 專案目標

- 建立 MuJoCo 上可重現的 Pick 任務 baseline。
- 目前訓練與評估流程是最小可運行骨架，尚未整合真正的 TD-MPC。

## 基本概念

- **環境 (Environment)**: 使用 Gymnasium 介面，環境 ID 為 `PushCube-v0`。
- **回合 (Episode)**: 每次 reset 後到終止條件或 timeout 的一段互動。
- **觀測 (Observation)**: 環境回傳的狀態資訊，可能是 dict 或 numpy array。
- **動作 (Action)**: 送進環境的控制輸入，通常是 Box 空間。
- **回饋 (Reward)**: 用於學習的數值訊號，目前仍是 placeholder 設定。

## 模型與訓練流程（目前為 placeholder）

- 現行 `train_pick.py` 使用一個簡單的 GPU MLP policy 產生動作，目的在於確認 CUDA 可用並完成訓練迴圈。
- 真正的 TD-MPC 目前未整合，`configs/algo/tdmpc.yaml` 僅為占位設定。
- `eval_pick.py` 使用同樣的 GPU policy 進行評估並可錄影。

## RL 訓練基礎知識

- **目標**: 最大化長期累積回饋（return）。
- **Policy**: 從觀測到動作的映射，訓練過程會逐步提升回饋。
- **探索 vs. 利用**: 訓練初期需要探索，後期偏向利用學到的策略。
- **終止條件**: 由成功條件或時間限制決定回合結束。
- **評估**: 固定 policy、不帶探索噪聲，觀察平均回報與成功率。

## 訓練流程與預期結果（目前為 placeholder）

- **流程**: reset 環境 → 產生 action → step → 累積回饋 → 回合結束後重置。
- **目前行為**: policy 是隨機/簡單 MLP，**不代表學習效果**。
- **預期輸出**:
  - 訓練期間會定期顯示 `step=... (placeholder eval)`。
  - 會寫入 `checkpoints/latest.json`（僅紀錄步數）。
  - 評估會輸出 `eval_summary.json`（包含平均回報與 checkpoint 資訊）。
- **真正的訓練結果**: 需等 TD‑MPC 整合後才會出現穩定提升的回報與成功率。

## 快速檢查與執行

### 安裝與環境

```bash
python -m venv .venv
source .venv/bin/activate
PIP_CONSTRAINT= pip install -r requirements.txt

mkdir -p third_party
cd third_party
git clone https://github.com/perezjln/gym-lowcostrobot.git
cd ..
```

### Smoke Test（確認環境與 GPU）

```bash
export MUJOCO_GL=egl
export XDG_CACHE_HOME="$(pwd)/.cache"
python scripts/smoke_test_env.py --env-id PushCube-v0
```

看到 `smoke_cuda: ... op=ok` 且 `episode 0 ok` 代表環境與 GPU 正常。

### 訓練（placeholder）

```bash
python scripts/train_pick.py --config configs/train_pick.yaml
```

### 評估（placeholder）

```bash
python scripts/eval_pick.py --ckpt checkpoints/latest.json --config configs/train_pick.yaml --episodes 1
```

## 設定檔導覽

- `configs/train_pick.yaml`: 訓練流程參數、log/checkpoint 路徑與 config 入口。
- `configs/env/lowcostrobot_pick.yaml`: 環境、reward、randomization 的設定。
- `configs/algo/tdmpc.yaml`: TD-MPC placeholder 設定（尚未接入訓練）。

## 主要輸出

- `logs/`: 訓練過程記錄
- `checkpoints/`: 目前僅寫入 `latest.json`（placeholder）
- `videos/`: eval 錄影輸出（可選）

## 常見問題

- **CUDA 不可用**: 請確認 driver、CUDA 與 PyTorch 安裝正確。
- **MuJoCo import 失敗**: 確認已安裝 `mujoco`，無頭環境請設定 `MUJOCO_GL=egl`。
- **shader cache 權限錯誤**: 設定 `XDG_CACHE_HOME` 到可寫入目錄。
