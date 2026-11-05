"""Synchronized Genesis wrapper benchmark; compare legacy calls without changing physics settings."""

import argparse
from dataclasses import replace
import json
from statistics import median
import time

import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mat_to_quat, quat_to_mat


def legacy_state(env):
    ent, root = env.robot, [env.root_link]
    return (ent.get_links_pos(root)[:, 0], quat_to_mat(ent.get_links_quat(root)[:, 0]),
            ent.get_links_vel(root)[:, 0], ent.get_links_ang(root)[:, 0],
            ent.get_dofs_position(env.dof_ids), ent.get_dofs_velocity(env.dof_ids))


def legacy_write(env, ids, p, R, v, w, q, qd):
    ent = env.robot
    ent.set_qpos(torch.cat((p, mat_to_quat(R)), -1), env.root_qs, ids, zero_velocity=False)
    ent.set_qpos(q, env.q_ids, ids, zero_velocity=False)
    w_local = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
    ent.set_dofs_velocity(torch.cat((v, w_local), -1), env.root_dofs, ids)
    ent.set_dofs_velocity(qd, env.dof_ids, ids)
    ent.control_dofs_position(q, env.dof_ids, ids)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument("--separate_substeps", action="store_true", help="measure the original four scene.step calls")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.reads) <= 0:
        parser.error("counts must be positive")
    dev = args.device
    torch.manual_seed(0)
    robot = get_robot("g1")
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
    layout = XiLayout(robot, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
    env = TrackingSim(replace(make_sim(robot), batch_substeps=not args.separate_substeps), args.num_envs, robot, motion, mdp, dev, eval_mode=True)
    ids = torch.arange(args.num_envs, device=dev)
    clips = ids % len(motion.names)
    frames = motion.clip_start[clips] + 100
    if not bool((frames + args.steps + 5 < motion.clip_end[clips]).all()):
        parser.error("--steps is too large for the selected motion clips")
    action = torch.zeros(args.num_envs, robot.num_joints, device=dev)
    sync = (lambda: torch.cuda.synchronize(dev)) if torch.device(dev).type == "cuda" else (lambda: None)

    def measure(fn, n, prepare=lambda: None):
        samples = []
        for _ in range(3):
            prepare()
            for _ in range(5):
                fn()
            sync()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            sync()
            samples.append(1000 * (time.perf_counter() - t0) / n)
        return median(samples)

    def reset():
        env.reset(ids, frames)
        env.done_eval.zero_(), env.failed_eval.zero_()

    reset()
    state = tuple(x.clone() for x in env.state())
    # Nonzero velocities and rotated poses exercise the free-joint frame conversion.
    for a, b in zip(state, legacy_state(env)):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
    result = {"device": dev, "num_envs": args.num_envs, "steps": args.steps,
              "physics_substeps_per_scene_step": env.sim_substeps,
              "scope": "eager simulator wrapper only; excludes neural updates and probe sweeps"}
    result["legacy_state_ms"] = measure(lambda: legacy_state(env), args.reads)
    result["packed_state_ms"] = measure(env.state, args.reads)
    result["legacy_reset_write_ms"] = measure(lambda: legacy_write(env, ids, *state), 20)
    result["packed_reset_write_ms"] = measure(lambda: env._write_state(ids, *state), 20)
    for a, b in zip(env.state(), state):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
    original_state, original_step = env.state, env.scene.step
    try:
        env.state = lambda: legacy_state(env)
        env.scene.step = lambda **kw: original_step(update_visualizer=True)
        result["legacy_control_step_ms"] = measure(lambda: env.step(action), args.steps, reset)
        old = tuple(x.clone() for x in env.state())
    finally:
        env.state, env.scene.step = original_state, original_step
    result["packed_control_step_ms"] = measure(lambda: env.step(action), args.steps, reset)
    for a, b in zip(env.state(), old):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
    # Step a random action as well: packed reads must agree with independently derived link state after integration.
    env.step(torch.randn_like(action))
    for a, b in zip(env.state(), legacy_state(env)):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
    result["state_and_rollout_equivalence"] = "passed"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
