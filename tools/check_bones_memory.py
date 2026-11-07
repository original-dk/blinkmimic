"""Validate full-library training memory on the target GPU, including MuJoCo/Warp allocations.

python tools/check_bones_memory.py --motion_dir /path/to/bones --log_dir logs/bones32_check --gpu 0

Runs 250 real training iterations (including graph captures, buffer fill and resets), without subsetting
the motion set. Polls nvidia-smi for whole-device usage; use an otherwise idle GPU. Saves memory_check.json
and returns nonzero if training fails or observed use reaches the budget. This is a smoke check, not a
claim that every possible long-run policy/data distribution has been exercised.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def gpu_memory(gpu):
    r = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    return tuple(int(x.strip()) for x in r.stdout.strip().split(','))  # MiB, whole device (also Warp allocations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion_dir', required=True)
    parser.add_argument('--motion_quat_order', choices=['wxyz', 'xyzw'], default='wxyz')
    parser.add_argument('--log_dir', default='logs/bones32_check')
    parser.add_argument('--gpu', default='0', help='nvidia-smi index or GPU UUID; mapped to cuda:0 for the child')
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--budget_gib', type=float, default=32.)
    args = parser.parse_args()
    if args.iterations < 250 or args.budget_gib <= 0:
        parser.error('use at least 250 iterations and a positive budget')
    repo = Path(__file__).resolve().parents[1]
    log = Path(args.log_dir).resolve()
    log.mkdir(parents=True, exist_ok=True)
    peak, total = gpu_memory(args.gpu)  # fail before starting training if monitoring is unavailable
    cmd = ['bash', str(repo/'scripts/train_bones32.sh'), str(log), '--motion_dir', str(Path(args.motion_dir).resolve()),
           '--motion_quat_order', args.motion_quat_order, '--iterations', str(args.iterations),
           '--log_every', '10', '--ckpt_every', str(args.iterations), '--ckpt_minutes', '0']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'PYTHON': sys.executable}
    print(f'Full-library GPU check; training output: {log/"memory_check_stdout.log"}', flush=True)
    with (log/'memory_check_stdout.log').open('w') as stdout:
        process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                used, total = gpu_memory(args.gpu)
                peak = max(peak, used)
                time.sleep(.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    limit = min(args.budget_gib, total/1024)
    events = EventAccumulator(str(log/'tensorboard'), size_guidance={'scalars': 0}).Reload() if (log/'tensorboard').is_dir() else None
    tags = events.Tags()['scalars'] if events is not None else []
    steps = events.Scalars('train/iteration') if 'train/iteration' in tags else []
    peaks = {tag: max(v.value for v in events.Scalars(tag)) for tag in
             ('memory/peak_allocated_gib', 'memory/peak_reserved_gib') if tag in tags}
    report = dict(command=cmd, returncode=process.returncode, observed_device_peak_gib=peak/1024,
                  budget_gib=args.budget_gib, device_total_gib=total/1024,
                  last_logged_iteration=steps[-1].step if steps else 0, torch_peaks=peaks,
                  passed=process.returncode == 0 and bool(steps) and steps[-1].step >= 250
                  and peak/1024 < limit and peaks.get('memory/peak_reserved_gib', limit+1) < limit)
    (log/'memory_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
