# Core
```
# train
PYTHON=python bash scripts/train.sh logs/run_s0   --seed 0 --yaw_degrees 90 --xy_range 0 0.1 --pi_ref_skip --iterations 300000 --ckpt_every 1000 --max_episode_steps 200

# eval
python -m blinkmimic.evaluateran   --run_dir logs/run_s0   --ckpt logs/run_s0/ckpt_300000.pt   --record logs/run_no_rel_s0/traj_ran.npz   --episodes 20 --yaw_degrees 90 --xy_range 0 0.1 --seed 0

# render
UJOCO_GL=egl python tools/render_ran.py   logs/run_no_rel_s0/traj_ran.npz logs/run_no_rel_s0/result_ran.mp4   --robot g1 --seconds 10
```