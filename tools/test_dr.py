"""Genesis DR integration check: parameter readback and isolation of partial resets."""

import argparse
from dataclasses import replace

import torch

from blinkmimic.dr import DomainRandomizer, XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.sensing import JointVelFilterCfg, SensingCfg

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
dev = args.device
torch.manual_seed(7)
robot = get_robot("g1")
robot = replace(robot, dr=replace(robot.dr, com_mass_range=(0.8, 1.2), link_mass_range=(0.9, 1.1),
                                pd_gain_scale=((tuple(range(robot.num_joints)), (0.8, 1.2), (0.7, 1.3)),),
                                armature_range=(0.8, 1.2), damping_range=(0.8, 1.2),
                                push_interval_s=(0.02, 0.02), push_velocity=(0.1,) * 6),
                sensing=SensingCfg(joint_vel_filter=JointVelFilterCfg(tau={".*": 0.01}, tau_scale_range=(0.8, 1.2)),
                                   delay_hold_prob=0.5))
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
layout = XiLayout(robot, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
env = TrackingSim(make_sim(robot), 3, robot, motion, mdp, dev, reset_noise=False, bin_fix=True)
r, ent = env.randomizer, env.robot
ids = torch.tensor([1], device=dev)
old_xi = env.xi.clone()
old_state = tuple(x.clone() for x in env.state())
frames = motion.clip_start[torch.zeros(1, dtype=torch.long, device=dev)] + 100
env.reset(ids, frames)
keep = torch.tensor([0, 2], device=dev)
torch.testing.assert_close(env.xi[keep], old_xi[keep], atol=0, rtol=0)
for a, b in zip(env.state(), old_state):
    torch.testing.assert_close(a[keep], b[keep], atol=0, rtol=0)
for a, b in zip(env.state(), motion.ref_state(frames)):
    torch.testing.assert_close(a[ids], b, atol=2e-5, rtol=1e-5)
for name, applied in (
    ("com", ent.get_links_COM([r.com_body_id])[:, 0] - r.default_com[:, r.com_body_id]),
    ("com_mass", ent.get_links_mass([r.com_body_id]) / r.default_mass[:, [r.com_body_id]]),
    ("link_mass", ent.get_links_mass(r.link_body_ids) / r.default_mass[:, r.link_body_ids]),
    ("kp", ent.get_dofs_kp(env.dof_ids) / env.kp),
    ("kd", ent.get_dofs_kv(env.dof_ids) / env.kd),
    ("armature", ent.get_dofs_armature(env.dof_ids) / env.armature),
):
    torch.testing.assert_close(layout.get(env.xi, name), applied)
    assert torch.isfinite(applied).all(), name
    print(f"[PASS] {name} readback")
assert (layout.get(env.xi, "joint_offset").abs() <= r.joint_offset_half).all()
assert not torch.equal(layout.get(env.xi, "joint_offset")[ids], layout.get(old_xi, "joint_offset")[ids])
assert torch.equal(layout.get(env.xi, "friction"), layout.get(old_xi, "friction"))
print("[PASS] partial reset isolation; per-episode offsets and persistent friction")
for _ in range(3):
    assert torch.isfinite(env.observe()).all()
    out = env.step(torch.zeros(3, robot.num_joints, device=dev))
    assert torch.isfinite(out["s1"]).all() and torch.isfinite(out["r"]).all()
print("[PASS] stateful sensing and velocity pushes")
unsupported = replace(robot, dr=replace(robot.dr, solref_geoms=".*"))
try:
    DomainRandomizer(unsupported, layout, env, eval_mode=False)
except ValueError as e:
    assert "solref" in str(e)
else:
    raise AssertionError("unsupported per-world solref must fail explicitly")
print("[PASS] unsupported solref rejected")
