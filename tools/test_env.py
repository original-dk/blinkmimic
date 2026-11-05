"""Check FK vs simulator, state round-trip, and reward on reference playback."""

import argparse
import time
from dataclasses import replace

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.rotations import quat_to_mat, rot_angle_sq

dev = args.device
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
t0 = time.time()
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
env = TrackingSim(make_sim(robot), args.num_envs, robot, motion, mdp, dev, eval_mode=True)
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
env.reset(torch.arange(args.num_envs, device=dev), frames)
env.refresh()
S = env.state()
ref = motion.ref_state(frames)
names = ["p", "R", "v", "w", "q", "qd"]
for n, a, b in zip(names, S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
# FK vs sim body frames
ids = [env.robot.get_link(b).idx_local for b in robot.tracked_bodies]
sp = env.robot.get_links_pos(ids)
sR = quat_to_mat(env.robot.get_links_quat(ids))
bp, bR, bv, bw = kin.forward(*S)
print("FK pos err", (bp - sp).norm(dim=-1).max().item())
print("FK rot err", rot_angle_sq(bR, sR).sqrt().max().item())
sv = torch.cat((env.robot.get_links_vel(ids), env.robot.get_links_ang(ids)), -1)
print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
torch.testing.assert_close(bp, sp, atol=2e-5, rtol=1e-5)
torch.testing.assert_close(bR, sR, atol=2e-5, rtol=1e-5)
torch.testing.assert_close(bv, sv[..., :3], atol=2e-5, rtol=1e-5)
torch.testing.assert_close(bw, sv[..., 3:], atol=2e-5, rtol=1e-5)
# open-loop reference playback via joint targets
rs = []
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    assert torch.isfinite(out["s1"]).all() and torch.isfinite(out["r"]).all()
    rs.append(out["r"].mean().item() / 0.1)
    if t % 25 == 0:
        print(t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())

# Grouping substeps must preserve the physical rollout, not merely advance the clock faster.
baseline = TrackingSim(replace(make_sim(robot), batch_substeps=False), args.num_envs, robot, motion, mdp, dev, eval_mode=True)
ids = torch.arange(args.num_envs, device=dev)
env.reset(ids, frames)
baseline.reset(ids, frames)
for _ in range(20):
    a = (motion.q[env.f + 1] - env.default_q) / env.alpha
    new, old = env.step(a), baseline.step(a)
    torch.testing.assert_close(new["s1"], old["s1"], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(new["r"], old["r"], atol=2e-5, rtol=1e-5)
    assert torch.equal(new["done"], old["done"])
print("[PASS] grouped vs separate physics substeps: 20-step state, reward and termination equivalence")
