#!/bin/bash
# Full FP32 reference library in RAM; a bounded GPU cache serves the unchanged reward/command heads.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: bash scripts/train_bones_cpu.sh <log_dir> --motion_dir <bones_dir> [train.py flags]}; shift
exec bash scripts/train.sh "$LOG_DIR" \
  --motion_cache_gib 6 --num_envs 4096 --buffer_steps 500 --wm_batch 2048 --pi_batch 2048 \
  --eprobe_share 0.15 --eprobe_envs 512 --eprobe_start 1000 --eprobe_every 250 --eprobe_decay 0.5 \
  --eprobe_async 0 --max_episode_steps 200 \
  --iterations 100000 --ckpt_every 1000 "$@"
