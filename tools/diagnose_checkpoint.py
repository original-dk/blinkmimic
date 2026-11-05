"""Controlled checkpoint rollouts with sensing noise and dynamics / encoder-offset DR isolated.

Uses the same reference starts in four conditions. This is a diagnostic sample, not the full-clip evaluation
protocol or the adaptive-reset training distribution. No optimizer updates or simulator parameters are changed.
"""

import argparse
import json
from pathlib import Path

import torch

import blinkmimic.networks as networks
import blinkmimic.rotations as rotations
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, checkpoint_physics, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import CONTACT_Z, MAX_STEP_REWARD, MDP
from blinkmimic.motion import CONTROL_DT, MotionLib
from blinkmimic.networks import Policy, WorldModel, policy_kwargs
from blinkmimic.robot import get_robot
from blinkmimic.rotations import so3_log


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="last.pt, including live policy and world model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["live", "ema"], default="live")
    parser.add_argument("--motion_dir", default=None)
    parser.add_argument("--sim_physics", choices=["mjlab", "genesis"], default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        parser.error("--num_envs and --steps must be positive")
    dev = args.device
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    saved = ck["args"]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rotations.ELEMENTWISE = saved.get("fused_rot", False)
    key = "pi_live" if args.policy == "live" else "pi"
    if key not in ck or "wm" not in ck:
        parser.error("use the full last.pt checkpoint with pi_live and wm weights")
    robot = get_robot(saved.get("robot", "g1"))
    if robot.sensing.stateful or robot.dr.push_interval_s is not None:
        parser.error("this comparison currently requires the G1 stateless sensing recipe without pushes")
    kin, layout = Kinematics(robot, dev), XiLayout(robot, dev)
    motion = MotionLib.from_dir(args.motion_dir or saved.get("motion_dir") or robot.motion_dir, robot, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
    widths = lambda name, default: tuple(int(n) for n in saved.get(name, default).split(","))  # noqa: E731
    K, H = saved["goal_frames"], saved["hist"]
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=H,
                enc_hidden=widths("enc_hidden", "2048,1024,512,256"), dec_hidden=widths("dec_hidden", "2048,1024,512,256"),
                **policy_kwargs(saved, robot)).to(dev)
    pi.load_state_dict(ck[key])
    pi.eval()
    wm = WorldModel(robot.num_joints, layout.dim, use_prev_action=robot.has_action_delay,
                    hidden=widths("wm_hidden", "2048,2048,2048"), structured=saved.get("wm_structured", False), dt=CONTROL_DT).to(dev)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    networks.AMP_DTYPE = torch.bfloat16 if saved.get("amp") == "bf16" and torch.device(dev).type == "cuda" else None
    ids = torch.arange(args.num_envs, device=dev)
    clips = ids % len(motion.names)
    span = (motion.clip_len[clips].cpu() - args.steps - 2).clamp(min=1)
    rng = torch.Generator().manual_seed(args.seed)
    frames = motion.clip_start[clips] + (torch.rand(args.num_envs, generator=rng) * span).long().to(dev)
    end = motion.clip_end[clips]
    feet = [robot.tracked_bodies.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    physics = checkpoint_physics(ck, args.sim_physics)
    report = {"iteration": ck["iteration"], "policy": args.policy, "sim_physics": physics,
              "training_sim_physics": saved.get("sim_physics") or "unrecorded",
              "amp": "bf16" if networks.AMP_DTYPE is not None else "none", "fused_rot": rotations.ELEMENTWISE,
              "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
              "scope": "matched reference starts, deterministic actions, no reset noise, no training updates",
              "dynamics_dr_includes_encoder_offsets": True,
              "observation_history": H, "exploration_noise": 0.0, "reset_noise": False,
              "starts": [{"clip": motion.names[int(c)], "frame": int(f)} for c, f in zip(clips.cpu(), frames.cpu())],
              "conditions": {}}
    initial, initial_xi = None, {}
    for name, dynamics, sensing in (("clean", False, False), ("sensing_only", False, True),
                                    ("dynamics_only", True, False), ("training_dr", True, True)):
        torch.manual_seed(args.seed)
        env = TrackingSim(make_sim(robot, physics), args.num_envs, robot, motion, mdp, dev, eval_mode=True, randomize=dynamics)
        # Keep dynamics / encoder offsets fixed while independently toggling the observation-noise branch.
        env.randomize = sensing
        torch.manual_seed(args.seed + 1)
        env.reset(ids, frames)
        state = tuple(x.clone() for x in env.state())
        if initial is None:
            initial = state
        else:
            for a, b in zip(initial, state):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
        if dynamics in initial_xi:
            torch.testing.assert_close(env.xi, initial_xi[dynamics], atol=0, rtol=0)
        else:
            initial_xi[dynamics] = env.xi.clone()
        torch.manual_seed(args.seed + 2)  # paired observation-noise draws, independent of DR reset draws
        sums = dict.fromkeys(("return", "track", "action_rate", "joint_limit", "body_pos", "body_ori"), 0.0)
        errors = {group: dict.fromkeys(("p_m", "rot_rad", "v_m_s", "w_rad_s", "q_rad", "qd_rad_s"), 0.0)
                  for group in ("all", "contact", "no_contact")}
        counts = dict.fromkeys(errors, 0)
        hist = None
        settled_return, settled_count = 0.0, 0
        for step in range(args.steps):
            alive = ~env.done_eval.clone()
            S = env.state()
            o = env.observe()
            hist = o[:, None].expand(-1, H, -1).clone() if hist is None else torch.cat((hist[:, 1:], o[:, None]), 1)
            a = pi(hist, motion.goal(env.f, K))
            predicted = wm.step(S, a, env.a_prev, layout.normalize(env.xi))
            out = env.step(a)
            actual = env.state()
            assert torch.isfinite(out["r"][alive]).all(), name
            terms = out["terms"]
            vals = {"return": out["r"] / MAX_STEP_REWARD, "track": terms["track"] / MAX_STEP_REWARD,
                    "action_rate": terms["action_rate"], "joint_limit": terms["joint_limit"],
                    "body_pos": terms["e_body_pos"], "body_ori": terms["e_body_ori"]}
            if step >= H:
                settled_return += torch.where(alive, vals["return"], 0.0).sum().item()
                settled_count += int(alive.sum())
            for k, value in vals.items():
                sums[k] += torch.where(alive, value, 0.0).sum().item()
            body_p = kin.forward(*actual)[0]
            contact = (body_p[:, feet, 2] < CONTACT_Z).any(-1)
            residuals = (predicted[0] - actual[0], so3_log(predicted[1] @ actual[1].transpose(-1, -2)),
                         *(predicted[i] - actual[i] for i in range(2, 6)))
            for group, mask in (("all", alive), ("contact", alive & contact), ("no_contact", alive & ~contact)):
                counts[group] += int(mask.sum())
                for k, residual in zip(errors[group], residuals):
                    assert torch.isfinite(residual[mask]).all(), (name, group, k)
                    mse = residual.square().mean(-1)
                    errors[group][k] += torch.where(mask, mse, 0.0).sum().item()
            env.f = torch.minimum(env.f, end - 1)
            if bool(env.done_eval.all()):
                break
        n = counts["all"]
        result = {"transitions": n, "n_failed": int(env.failed_eval.sum()),
                  "mean_step_return": sums["return"] / n,
                  "settled_step_return": settled_return / settled_count if settled_count else None,
                  "settled_transitions": settled_count, "tracking_contribution": sums["track"] / n,
                  "action_rate_contribution": mdp.cfg.w_action_rate * CONTROL_DT / MAX_STEP_REWARD * sums["action_rate"] / n,
                  "joint_limit_contribution": mdp.cfg.w_joint_limit * CONTROL_DT / MAX_STEP_REWARD * sums["joint_limit"] / n,
                  "body_pos_rmse_cm": 100 * (sums["body_pos"] / n)**0.5,
                  "body_ori_rmse_deg": 57.2958 * (sums["body_ori"] / n)**0.5,
                  "wm_one_step_rmse": {group: {k: (v / counts[group])**0.5 for k, v in value.items()} if counts[group] else None
                                       for group, value in errors.items()}, "wm_transition_counts": counts}
        assert abs(result["mean_step_return"] - sum(result[k] for k in
                   ("tracking_contribution", "action_rate_contribution", "joint_limit_contribution"))) < 1e-5
        report["conditions"][name] = result
        print(f"{name}: return {result['mean_step_return']:.4f} | track {result['tracking_contribution']:.4f} "
              f"rate {result['action_rate_contribution']:.4f} limit {result['joint_limit_contribution']:.4f} "
              f"| failed {result['n_failed']}/{args.num_envs} | WM v {result['wm_one_step_rmse']['all']['v_m_s']:.4f} "
              f"qd {result['wm_one_step_rmse']['all']['qd_rad_s']:.4f}", flush=True)
    path = Path(args.out) if args.out else Path(args.ckpt).with_name("checkpoint_diagnostics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[PASS] matched starts / dynamics and reward decomposition; report: {path}")


if __name__ == "__main__":
    main()
