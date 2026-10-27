#!/usr/bin/env bash
# MOT20 run on a Chameleon GPU node (CC-Ubuntu22.04-CUDA image, user "cc").
# Settings are LOCKED from MOT17 (configs/mot20.yaml): nothing is tuned on MOT20.
#
#   python3 -m zipfile -e ondemand-depth-mot.zip .
#   tmux new -s mot
#   bash ondemand-depth-mot/scripts/chameleon_mot20.sh
#
# Output: ~/results_mot20.zip  - copy it home with scp.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU visible"; exit 1; }

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip unzip zip wget tmux >/dev/null

echo "== Python environment"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
# CUDA 12.6 wheels work with the driver on Chameleon's CUDA image
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"
python tests/test_pipeline.py | tail -1

echo "== MOT20 (about 5 GB)"
if [ ! -f data/MOT20/train/MOT20-01/seqinfo.ini ]; then
  mkdir -p data
  wget -q --show-progress -O data/MOT20.zip https://motchallenge.net/data/MOT20.zip
  unzip -q data/MOT20.zip -d data && rm data/MOT20.zip
fi
CFG="--config configs/mot20.yaml"
python -m odmot $CFG check
python -m odmot $CFG estimate

echo "== Caching (detector, then depth)"
python -m odmot $CFG cache-det
python -m odmot $CFG cache-depth

echo "== Experiments (all frames: nothing was tuned on MOT20)"
python -m odmot $CFG sweep --split full
python -m odmot $CFG analyze --split full
python -m odmot $CFG demo --seq MOT20-02 --frames 150 --policy od_margin_thr0.7 \
  --video results/demo_MOT20-02.mp4 || echo "(demo failed - not critical)"

echo "== Packaging"
nvidia-smi --query-gpu=name --format=csv,noheader > results/gpu_used.txt
cd "$ROOT" && zip -qr ~/results_mot20.zip cache results configs/mot20.yaml
ls -lh ~/results_mot20.zip
echo "DONE. Copy ~/results_mot20.zip to your laptop."
