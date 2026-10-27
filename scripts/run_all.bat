@echo off
REM Full experiment on a Windows laptop. Run from the repo root with the venv active.
REM Caching steps are resumable: if the laptop sleeps or you stop it, just run again.
python -m odmot check || exit /b 1
python -m odmot estimate
python -m odmot cache-det || exit /b 1
python -m odmot cache-depth || exit /b 1
python -m odmot sweep --split train_half || exit /b 1
python -m odmot sweep --split val_half || exit /b 1
python -m odmot analyze --split val_half || exit /b 1
python -m odmot demo --seq MOT17-09 --frames 150 --policy od_margin_thr0.7 --video results\demo_MOT17-09.mp4
