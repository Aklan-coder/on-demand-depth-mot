#!/usr/bin/env bash
# One-shot GPU run on a Chameleon Cloud node (Ubuntu + CUDA image, user "cc").
#
# On the node, from the folder that contains ondemand-depth-mot.zip:
#   unzip -q ondemand-depth-mot.zip || python3 -m zipfile -e ondemand-depth-mot.zip .
#   bash ondemand-depth-mot/scripts/chameleon_run.sh
#
# Run it inside tmux so an SSH disconnect does not stop it:
#   tmux new -s mot      (detach: Ctrl+B then D;  re-attach: tmux attach -t mot)
#
# Output: ~/results_gpu.zip  (caches + results + plots + demo video) - copy it home with scp.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo "== Project: $ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU visible"; exit 1; }

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip unzip zip wget tmux >/dev/null

echo "== Python environment"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q torch torchvision            # Linux PyPI wheels include CUDA
pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"
python tests/test_pipeline.py | tail -1

echo "== MOT17"
if [ ! -f data/MOT17/train/MOT17-02-FRCNN/seqinfo.ini ]; then
  mkdir -p data
  wget -q --show-progress -O data/MOT17.zip https://motchallenge.net/data/MOT17.zip
  unzip -q data/MOT17.zip -d data && rm data/MOT17.zip
fi
python -m odmot check

echo "== GPU config"
python - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/default.yaml"))
c["detector"]["device"] = "0"     # ultralytics: GPU index
c["depth"]["device"] = "cuda"     # torch device
yaml.safe_dump(c, open("configs/gpu.yaml", "w"), sort_keys=False)
EOF
CFG="--config configs/gpu.yaml"
python -m odmot $CFG estimate

echo "== Caching (detector, then depth)"
python -m odmot $CFG cache-det
python -m odmot $CFG cache-depth

echo "== Experiments"
python -m odmot $CFG sweep --split train_half
python -m odmot $CFG sweep --split val_half
python -m odmot $CFG analyze --split val_half
python -m odmot $CFG demo --seq MOT17-09 --frames 150 --policy od_margin_thr0.7 \
  --video results/demo_MOT17-09.mp4 || echo "(demo failed - not critical)"

echo "== Packaging"
nvidia-smi --query-gpu=name --format=csv,noheader > results/gpu_used.txt
cd "$ROOT" && zip -qr ~/results_gpu.zip cache results configs/gpu.yaml
ls -lh ~/results_gpu.zip
echo "DONE. Copy ~/results_gpu.zip to your laptop, then delete the Chameleon lease."
