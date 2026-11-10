
## Core
```
# train
PYTHON=python bash scripts/train.sh logs/run_history   --seed 0 --iterations 300000 --ckpt_every 1000 --vel_lr 1e-3 --vel_mix_rate 0.0001

# eval
python -m blinkmimic.evaluate   --run_dir logs/run_s0   --ckpt logs/run_history/ckpt_300000.pt   --record logs/run_history/traj_ran.npz

# render
MUJOCO_GL=egl python tools/render.py   logs/run_history/traj_ran.npz logs/run_history/result_ran.mp4   --robot g1 --seconds 60
```