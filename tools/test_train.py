"""Small Genesis training / resume integration run; checks updates and saved optimizer state."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
(root / "logs").mkdir(exist_ok=True)
run = Path(tempfile.mkdtemp(prefix="genesis_test_", dir=root / "logs"))
cmd = [sys.executable, "-m", "blinkmimic.train", "--device", args.device, "--eager", "--amp", "none",
       "--num_envs", "2", "--buffer_steps", "32", "--warmup_steps", "12", "--wm_window", "2",
       "--wm_batches", "1", "--wm_batch", "4", "--pi_batch", "4", "--horizon", "2", "--goal_frames", "2",
       "--hist", "2", "--wm_hidden", "32,32", "--enc_hidden", "32,16", "--dec_hidden", "32,16",
       "--critic_epochs", "1", "--critic_minibatches", "1", "--wm_only_iters", "0", "--log_every", "1",
       "--ckpt_every", "1", "--ckpt_minutes", "0", "--log_dir", str(run), "--bin_fix", "--pi_ema", "0.995",
       "--wm_structured", "--wm_mirror", "0.5", "--pi_ref_skip", "--contact_reward", "0.25", "--ref_contact_fix",
       "--pi_opt", "muon", "--wm_opt", "muon", "--fail_reset", "0.3"]
env = dict(os.environ, OMP_NUM_THREADS="4")
subprocess.run(cmd + ["--iterations", "3"], cwd=root, env=env, check=True)
first = torch.load(run / "ckpt_000001.pt", map_location="cpu", weights_only=False)
last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
assert last["iteration"] == 3 and last["sim_steps"] > first["sim_steps"]
for name in ("pi_live", "wm"):
    assert all(torch.isfinite(x).all() for x in last[name].values())
    changed = any(not torch.equal(first[name][k], v) for k, v in last[name].items() if k.endswith("weight"))
    assert changed, f"{name} weights did not update"
print("[PASS] world-model and policy weights updated; checkpoint tensors finite", flush=True)
subprocess.run(cmd + ["--iterations", "4", "--resume", str(run / "last.pt")], cwd=root, env=env, check=True)
resumed = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
assert resumed["iteration"] == 4 and resumed["sim_steps"] > last["sim_steps"]
rows = list(csv.DictReader((run / "train_log.csv").open()))
assert [int(r["iteration"]) for r in rows] == [1, 2, 3, 4]
for r in rows:
    for key in ("wm_loss", "wm_err8", "pi_loss", "v_loss", "pi_gradnorm", "imag_rew", "sim_rew"):
        assert math.isfinite(float(r[key])), (key, r[key])
    assert float(r["nonfinite"]) == 0
print(f"[PASS] four training iterations including resume; finite losses and gradients; artifacts: {run}")
