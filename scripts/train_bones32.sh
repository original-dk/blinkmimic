#!/bin/bash
# Full motion library, conservative single-GPU settings for a 32 GiB card. Extra flags override these defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: bash scripts/train_bones32.sh <log_dir> --motion_dir <bones_dir> [train.py flags]}; shift
exec bash scripts/train.sh "$LOG_DIR" \
  --num_envs 2048 --buffer_steps 500 --wm_batch 1024 --pi_batch 1024 \
  --eprobe_share 0 --eprobe_async 0 --max_episode_steps 200 \
  --iterations 320000 --ckpt_every 4000 "$@"
