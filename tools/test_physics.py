"""Compare Genesis profiles with a MuJoCo CPU reference using the original G1 PD / solver recipe.

This checks deterministic short rollouts, not training convergence or mjlab / MuJoCo-Warp GPU parity.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, checkpoint_physics, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mat_to_quat


def reference_model(robot):
    spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 0, 1
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), friction=(1, 0.005, 0.0001))
    for name, d in zip(robot.joint_names, robot.drives):
        spec.joint(name).armature = d.armature
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[d.stiffness] + [0] * 9, biasprm=[0, -d.stiffness, -d.damping] + [0] * 7,
                          forcelimited=True, forcerange=(-d.effort_limit, d.effort_limit))
    model = spec.compile()
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations, model.opt.ls_iterations, model.opt.ccd_iterations = 100, 50, 35
    model.opt.tolerance = 1e-6
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/physics_comparison.json")
    args = parser.parse_args()
    dev = args.device
    robot = get_robot("g1")
    # Old checkpoints must retain their historical physics unless the caller explicitly overrides it.
    assert checkpoint_physics({"args": {}}) == "genesis"
    assert checkpoint_physics({"args": {"sim_physics": "mjlab"}}) == "mjlab"
    assert checkpoint_physics({"args": {}}, "mjlab") == "mjlab"
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
    layout = XiLayout(robot, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
    frames = motion.clip_start[torch.arange(4, device=dev) * len(motion.names) // 4] + 100
    model = reference_model(robot)
    q_ids = [model.jnt_qposadr[model.joint(n).id] for n in robot.joint_names]
    v_ids = [model.jnt_dofadr[model.joint(n).id] for n in robot.joint_names]
    report = {"reference": "MuJoCo CPU, original PD and implicitfast / Newton settings", "profiles": {}}
    for physics in ("genesis", "mjlab"):
        env = TrackingSim(make_sim(robot, physics), 4, robot, motion, mdp, dev, eval_mode=True)
        ids = torch.arange(4, device=dev)
        params = env.scene.sim.rigid_solver.get_sol_params()
        result = {"geom_timeconsts": torch.unique(params[:, 0]).cpu().tolist()}
        expected = 0.02 if physics == "mjlab" else 0.01
        torch.testing.assert_close(params[:, 0], torch.full_like(params[:, 0], expected))
        for scenario, height in (("air", 3.0), ("floor", 0.0)):
            env.reset(ids, frames)
            p, R, v, w, q, qd = (x.clone() for x in env.state())
            p[:, 2] += height
            env._write_state(ids, p, R, v, w, q, qd)
            pose = torch.cat((p, mat_to_quat(R), q), -1).cpu().numpy()
            vel = torch.cat((v, (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1), qd), -1).cpu().numpy()
            data = [mujoco.MjData(model) for _ in range(4)]
            for i, d in enumerate(data):
                d.qpos[:7], d.qpos[q_ids], d.qvel[:6], d.qvel[v_ids] = pose[i, :7], pose[i, 7:], vel[i, :6], vel[i, 6:]
                mujoco.mj_forward(model, d)
            samples = []
            for step in range(20):
                target = motion.q[frames + step + 1]
                action = (target - env.default_q) / env.alpha
                env.step(action)
                for i, d in enumerate(data):
                    d.ctrl[:] = target[i].cpu().numpy()
                    for _ in range(4):
                        mujoco.mj_step(model, d)
                if step in (0, 4, 19):
                    state = env.state()
                    q_err = state[4].cpu().numpy() - np.stack([d.qpos[q_ids] for d in data])
                    qd_err = state[5].cpu().numpy() - np.stack([d.qvel[v_ids] for d in data])
                    p_err = state[0].cpu().numpy() - np.stack([d.qpos[:3] for d in data])
                    samples.append({"step": step + 1, "q_rmse_rad": float(np.sqrt(np.mean(q_err**2))),
                                    "qd_rmse_rad_s": float(np.sqrt(np.mean(qd_err**2))), "root_p_max_m": float(np.abs(p_err).max())})
            result[scenario] = samples
            if physics == "mjlab":
                assert max(s["q_rmse_rad"] for s in samples) < 1e-4, samples
                assert max(s["qd_rmse_rad_s"] for s in samples) < 1e-3, samples
                assert max(s["root_p_max_m"] for s in samples) < 1e-4, samples
        report["profiles"][physics] = result
        print(json.dumps({physics: result}, indent=2), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[PASS] aligned short-rollout physics and checkpoint profile defaults; report: {path}")


if __name__ == "__main__":
    main()
