#!/usr/bin/env bash
set -euo pipefail

# System deps for MuJoCo + headless rendering
apt-get update -y
apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libosmesa6

# Python deps
pip install -r requirements.txt

# Clone upstream env if missing
if [ ! -d "third_party/Sim-LeRobotHackathon" ]; then
  mkdir -p third_party
  git clone https://github.com/xxx/Sim-LeRobotHackathon.git third_party/Sim-LeRobotHackathon
fi

echo "Setup complete. You may need: export MUJOCO_GL=egl"
