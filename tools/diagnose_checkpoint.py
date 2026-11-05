"""Controlled checkpoint rollouts isolating DR, exploration and reset conditions.

The default four conditions use matched reference starts. Optional conditions add training exploration,
reset perturbations and the saved adaptive sampler. Weights and normalizers stay frozen; these samples
are diagnostic comparisons rather than the full-clip evaluation or an exact replay of training.
"""

import argparse
import json
from pathlib import Path

import torch

import blinkmimic.networks as networks
import blinkmimic.rotations as rotations
from blinkmimic import env as simulator
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import CONTACT_Z, MAX_STEP_REWARD, MDP, unflatten_state
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
    parser.add_argument("--conditions", nargs="+", choices=["clean", "sensing_only", "dynamics_only", "training_dr",
                                                           "exploration", "reset_noise", "curriculum"],
                        default=["clean", "sensing_only", "dynamics_only", "training_dr"])
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="override the curriculum rollout episode cap; default is the training cap")
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        parser.error("--num_envs and --steps must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max_episode_steps must be positive")
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
    # The script can run against the original mjlab checkout via PYTHONPATH=src, using that checkout's dependencies.
    backend = "mjlab" if type(make_sim(robot)).__module__.startswith("mjlab.") else "genesis"
    if backend == "mjlab" and args.sim_physics is not None:
        parser.error("--sim_physics selects a Genesis parameter profile; omit it when running against the original mjlab checkout")
    physics = simulator.checkpoint_physics(ck, args.sim_physics) if backend == "genesis" else None
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
    report = {"iteration": ck["iteration"], "policy": args.policy, "sim_backend": backend, "sim_physics": physics,
              "training_sim_physics": saved.get("sim_physics") or "unrecorded",
              "amp": "bf16" if networks.AMP_DTYPE is not None else "none", "fused_rot": rotations.ELEMENTWISE,
              "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
              "scope": "frozen checkpoint; controlled noise / sampler conditions, no training updates",
              "dynamics_dr_includes_encoder_offsets": True,
              "observation_history": H,
              "starts": [{"clip": motion.names[int(c)], "frame": int(f)} for c, f in zip(clips.cpu(), frames.cpu())],
              "conditions": {}}
    configs = (("clean", False, False, 0.0, False, False),
               ("sensing_only", False, True, 0.0, False, False),
               ("dynamics_only", True, False, 0.0, False, False),
               ("training_dr", True, True, 0.0, False, False),
               ("exploration", True, True, saved.get("action_noise", 0.1), False, False),
               ("reset_noise", True, True, saved.get("action_noise", 0.1), True, False),
               ("curriculum", True, True, saved.get("action_noise", 0.1), True, True))
    initial, initial_xi = None, {}
    for name, dynamics, sensing, action_noise, reset_noise, auto_reset in configs:
        if name not in args.conditions:
            continue
        torch.manual_seed(args.seed)
        cap = args.max_episode_steps or saved.get("max_episode_steps", 500)
        sim_cfg = make_sim(robot, physics) if backend == "genesis" else make_sim(robot)
        env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev,
                          eval_mode=not reset_noise, randomize=dynamics, reset_noise=reset_noise,
                          max_episode_steps=cap, gamma=saved.get("gamma", 0.99),
                          fail_reset=saved.get("fail_reset", 0.0) if auto_reset else 0.0, bin_fix=saved.get("bin_fix", False))
        # Evaluation bookkeeping can be used with a perturbed initial state; only the curriculum condition auto-resets.
        env.eval_mode = not auto_reset
        if not auto_reset:
            env.max_episode_steps = 10**9
        env.randomize = sensing
        if auto_reset:
            ctrl = ck.get("ctrl") or {}
            bins = ctrl.get("env_bins")
            if bins is None:
                parser.error("curriculum requires a full last.pt with ctrl.env_bins")
            for key in ("bin_G", "bin_fail", "bin_w_avg"):
                source = bins[key].to(dev)
                if source.shape != getattr(env, key).shape:
                    parser.error("saved curriculum does not match the selected motion data")
                getattr(env, key).copy_(source)
            if ctrl.get("focus_q") is not None:
                env.set_focus(ctrl["focus_q"].to(dev), float(ctrl["focus_share"]))
        torch.manual_seed(args.seed + 1)
        env.reset(ids, None if auto_reset else frames)
        if not auto_reset:
            if dynamics in initial_xi:
                # Keep encoder offsets paired when reset-pose noise consumes additional random draws.
                sl = layout.slices["joint_offset"]
                env.xi[:, sl] = initial_xi[dynamics][:, sl]
                torch.testing.assert_close(env.xi, initial_xi[dynamics], atol=0, rtol=0)
            else:
                initial_xi[dynamics] = env.xi.clone()
            if not reset_noise:
                state = tuple(x.clone() for x in env.state())
                if initial is None:
                    initial = state
                else:
                    for a, b in zip(initial, state):
                        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
        case_config = {"dynamics_dr": dynamics, "sensing_noise": sensing, "action_noise": action_noise,
                       "reset_noise": reset_noise, "auto_reset": auto_reset, "episode_cap": env.max_episode_steps,
                       "fail_reset": env.fail_reset, "bin_fix": env.bin_fix,
                       "initial_frames": env.f.cpu().tolist(), "initial_focus_share": env.focus_share,
                       "sampler": "saved adaptive sampler" if auto_reset else "matched reference starts"}
        torch.manual_seed(args.seed + 2)
        sums = dict.fromkeys(("return", "track", "action_rate", "joint_limit", "body_pos", "body_ori"), 0.0)
        errors = {group: dict.fromkeys(("p_m", "rot_rad", "v_m_s", "w_rad_s", "q_rad", "qd_rad_s"), 0.0)
                  for group in ("all", "contact", "no_contact")}
        counts = dict.fromkeys(errors, 0)
        hist = None
        settled_return, settled_count = 0.0, 0
        episode_ends, terminations = 0, 0
        for _ in range(args.steps):
            alive = torch.ones_like(env.done_eval) if auto_reset else ~env.done_eval.clone()
            S = env.state()
            o = env.observe()
            repeated = o[:, None].expand(-1, H, -1)
            hist = repeated.clone() if hist is None else torch.where((env.ep_step == 0)[:, None, None], repeated,
                                                                   torch.cat((hist[:, 1:], o[:, None]), 1))
            a = pi(hist, motion.goal(env.f, K))
            if action_noise:
                a = torch.clamp(a + action_noise * torch.randn_like(a), -20, 20)
            fresh = env.ep_step < H
            predicted = wm.step(S, a, env.a_prev, layout.normalize(env.xi))
            out = env.step(a)
            # step() may already have reset ended episodes. Score the transition's pre-reset s1, as the replay buffer does.
            actual = unflatten_state(out["s1"])
            episode_ends += int((out["done"] & alive).sum())
            terminations += int((out["term"] & alive).sum())
            assert torch.isfinite(out["r"][alive]).all(), name
            terms = out["terms"]
            vals = {"return": out["r"] / MAX_STEP_REWARD, "track": terms["track"] / MAX_STEP_REWARD,
                    "action_rate": terms["action_rate"], "joint_limit": terms["joint_limit"],
                    "body_pos": terms["e_body_pos"], "body_ori": terms["e_body_ori"]}
            settled = alive & ~fresh
            settled_return += torch.where(settled, vals["return"], 0.0).sum().item()
            settled_count += int(settled.sum())
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
            if not auto_reset:
                env.f = torch.minimum(env.f, end - 1)
            if not auto_reset and bool(env.done_eval.all()):
                break
        n = counts["all"]
        result = {"config": case_config, "transitions": n, "n_failed": None if auto_reset else int(env.failed_eval.sum()),
                  "episode_ends": episode_ends, "episode_terminations": terminations,
                  "term_fraction": terminations / episode_ends if episode_ends else None,
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
              f"| terminations {terminations}/{episode_ends} | WM v {result['wm_one_step_rmse']['all']['v_m_s']:.4f} "
              f"qd {result['wm_one_step_rmse']['all']['qd_rad_s']:.4f}", flush=True)
    path = Path(args.out) if args.out else Path(args.ckpt).with_name("checkpoint_diagnostics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[PASS] selected condition controls, pre-reset transition scoring and reward decomposition; report: {path}")


if __name__ == "__main__":
    main()
