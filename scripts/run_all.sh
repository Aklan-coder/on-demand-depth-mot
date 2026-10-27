#!/usr/bin/env bash
# Full experiment (Linux / macOS / Git-Bash). Run from the repo root with the venv active.
# Caching steps are resumable: if interrupted, just run again.
set -e
python -m odmot check
python -m odmot estimate
python -m odmot cache-det
python -m odmot cache-depth
python -m odmot sweep --split train_half
python -m odmot sweep --split val_half
python -m odmot analyze --split val_half
python -m odmot demo --seq MOT17-09 --frames 150 --policy od_margin_thr0.7 --video results/demo_MOT17-09.mp4
