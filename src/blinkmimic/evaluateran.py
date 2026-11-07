"""Evaluate from random motions, frames and robot headings, keeping original actor commands.

python -m blinkmimic.evaluateran --run_dir logs/<run> --ckpt <checkpoint> --record traj.npz
Each rollout samples a clip, a frame before its last frame, a robot yaw offset, and an XY translation. Expert motion,
commands and observation references are untouched, as in training. Ghost alignment is display-only;
render_ran.py uses the saved initial frame and yaw to rotate and translate the reference to the robot reset pose.
"""

from __future__ import annotations

import math

import torch

from .env import sample_reset_xy
from .motion import MotionLib
from .rotations import mat_to_quat, mv3, so3_exp


@torch.no_grad()
def sample_starts(motion: MotionLib, n: int, half: float, seed: int):
    """Uniform clip, then uniform frame with at least one transition, then independent robot yaw.

    A private generator keeps the same evaluation starts across checkpoints and does not depend on
    policy initialization or simulation RNG. The final frame cannot start a transition.
    """
    if n < 1 or not 0 <= half <= math.pi:
        raise ValueError("rollout count must be positive and yaw half-range must be in [0, pi]")
    valid = (motion.clip_len >= 2).nonzero().flatten()
    if not len(valid):
        raise ValueError("evaluation needs at least one motion with two frames")
    dev = motion.root_p.device
    rng = torch.Generator(device=dev).manual_seed(seed)
    clips = valid[torch.randint(len(valid), (n,), device=dev, generator=rng)]
    frames = motion.clip_start[clips] + (torch.rand(n, device=dev, generator=rng)
                                       * (motion.clip_len[clips] - 1)).long()
    yaw = torch.empty(n, device=dev).uniform_(-half, half, generator=rng)
    return clips, frames, yaw


@torch.no_grad()
def reset_random_yaw(env, frames: torch.Tensor, yaw: torch.Tensor, pos_offset: torch.Tensor | None = None):
    """Reset from the untouched expert frame, rotating and translating only the robot as in training.

    Reference frames, command generation and observation references remain unchanged. Save the
    robot's randomized reset pose so reward compares each trajectory in its own initial frame.
    """
    ids = torch.arange(len(frames), device=frames.device)
    env.reset(ids, frames)
    p, R, v, w, _, _ = env.motion.ref_state(frames)
    rot = torch.zeros_like(p)
    rot[:, 2] = yaw
    Y = so3_exp(rot)
    if pos_offset is not None:
        p = p + pos_offset
    pose = torch.cat((p, mat_to_quat(Y @ R)), -1)
    env.robot.write_root_link_pose_to_sim(torch.cat((p + env.scene.env_origins, pose[:, 3:]), -1), env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((mv3(Y, v), mv3(Y, w)), -1), env_ids=ids)
    env.init_root.copy_(pose)



def main():
    import argparse
    import glob
    import json
    import os
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="g1", help="robot package under blinkmimic/ (g1)")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
    parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
    parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
    parser.add_argument("--motion_cache_gib", type=float, default=0, help="FP32 CPU library with bounded GPU cache; 0 disables")
    parser.add_argument("--motion_cache_dir", type=str, default=None, help="FP32 SSD cache (default: <run_dir>/motion_cache)")
    parser.add_argument("--motion_quat_order", choices=["wxyz", "xyzw"], default="wxyz",
                        help="NPZ root quaternion order (default: wxyz); CSV keeps the robot's format")
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    parser.add_argument("--dr_from", type=str, default=None, help="evaluate under this robot's training randomization ('self': own)")
    parser.add_argument("--dr_parts", type=str, default="all", choices=["all", "dynamics", "sensing"],
                        help="with --dr_from: take only its simulator randomization or only its sensing")
    parser.add_argument("--episodes", type=int, default=1, help="independent random motion/frame/yaw rollouts")
    parser.add_argument("--repeats", type=int, default=1, help="multiply --episodes by this count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xy_range", type=float, nargs=2, default=(0.0, 0.1), metavar=("LOW", "HIGH"),
                        help="independent world-X/Y reset translation interval in meters (0 0 disables)")
    parser.add_argument("--yaw_degrees", type=float, default=90.0, help="uniform yaw offset half-range (degrees; 0 disables)")
    args = parser.parse_args()
    if not 0 <= args.yaw_degrees <= 180:
        parser.error("--yaw_degrees must be between 0 and 180")
    if not all(math.isfinite(x) for x in args.xy_range) or args.xy_range[0] > args.xy_range[1]:
        parser.error("--xy_range requires finite LOW <= HIGH")
    if args.repeats < 1 or args.episodes < 1:
        parser.error("--episodes and --repeats must be positive")
    if not math.isfinite(args.motion_cache_gib) or args.motion_cache_gib < 0:
        parser.error("--motion_cache_gib must be finite and nonnegative")

    import dataclasses
    import re

    import numpy as np
    import torch

    import blinkmimic.networks as networks
    from blinkmimic.dr import XiLayout
    from blinkmimic.env import TrackingSim, make_sim
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.mdp import MAX_STEP_REWARD, MDP
    from blinkmimic.motion import MotionLib
    from blinkmimic.networks import Policy, policy_kwargs
    from blinkmimic.robot import RobotSpec, get_robot


    def transfer_dr(target: RobotSpec, source: RobotSpec, parts: str = "all") -> RobotSpec:
        """``source``'s randomization and/or sensing on ``target``. PD-gain ranges are given per actuator group,
        so each target group takes the range of the source entry that covers most of its joints."""
        if parts == "sensing":
            return dataclasses.replace(target, sensing=source.sensing)
        if parts == "dynamics":
            return dataclasses.replace(transfer_dr(target, source), sensing=target.sensing)
        if target.name == source.name:
            return target
        src_acts = source.robot_cfg().articulation.actuators
        tgt_acts = target.robot_cfg().articulation.actuators
        joints = lambda act, names: {n for n in names if any(re.fullmatch(p, n) for p in act.target_names_expr)}  # noqa: E731
        entries = [(set().union(*(joints(src_acts[i], source.joint_names) for i in ids)), kp, kd) for ids, kp, kd in source.dr.pd_gain_scale]
        groups: dict[int, list[int]] = {}
        for i, act in enumerate(tgt_acts):
            tj = joints(act, target.joint_names)
            best = max(range(len(entries)), key=lambda e: len(tj & entries[e][0]))
            groups.setdefault(best, []).append(i)
        pd = tuple((tuple(ids), entries[e][1], entries[e][2]) for e, ids in sorted(groups.items()))
        dr = dataclasses.replace(source.dr, pd_gain_scale=pd)
        return dataclasses.replace(target, dr=dr, sensing=source.sensing, com_range=source.com_range)


    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    dev = "cuda:0"
    robot = get_robot(args.robot)
    if args.dr_from:
        robot = transfer_dr(robot, robot if args.dr_from == "self" else get_robot(args.dr_from), args.dr_parts)
        print("pd_gain_scale on this robot:", robot.dr.pd_gain_scale)
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev, args.motion_quat_order,
                                cache_gib=args.motion_cache_gib,
                                cache_dir=args.motion_cache_dir or os.path.join(args.run_dir, 'motion_cache'))
    C = len(motion.names)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    N = args.episodes * args.repeats
    clip_of, start_frames, yaw = sample_starts(motion, N, math.radians(args.yaw_degrees), args.seed)
    motion.prepare(start_frames)
    pos_rng = torch.Generator(device=motion.root_p.device).manual_seed(args.seed + 1)
    pos_offset = sample_reset_xy(motion.root_p[start_frames], tuple(args.xy_range), pos_rng)


    @torch.no_grad()
    def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
        a_ = ck["args"]
        K, Hh = a_["goal_frames"], a_["hist"]
        networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
        widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
        pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh, enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                    dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
        pi.load_state_dict(ck["pi"])
        pi.eval()
        env.done_eval[:] = False
        env.failed_eval[:] = False
        reset_random_yaw(env, start_frames, yaw, pos_offset)
        clip_end = motion.clip_end[clip_of]
        ret = torch.zeros(N, device=dev)
        root_sq_sum = torch.zeros(N, device=dev)
        root_sq_max = torch.zeros(N, device=dev)
        root_sq_final = torch.zeros(N, device=dev)
        steps = torch.zeros(N, device=dev)
        hist = None
        traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
        while not bool(env.done_eval.all()):
            motion.prepare(env.f, ahead=K+1, extra=env.init_f)
            S = env.state()
            o = env.observe()
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(env.f, K))
            alive = ~env.done_eval.clone()
            if record:
                from blinkmimic.rotations import mat_to_quat

                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            out = env.step(a)
            ret += torch.where(alive, out["r"], 0.0)
            # The reward error is in the two initial root frames. Its norm equals the distance
            # to the fixed, yaw-rotated ghost, including when the initial expert pose is tilted.
            e_root = out["terms"]["e_root_pos"]
            root_sq_sum += torch.where(alive, e_root, 0.0)
            root_sq_max = torch.maximum(root_sq_max, torch.where(alive, e_root, 0.0))
            root_sq_final = torch.where(alive, e_root, root_sq_final)
            steps += alive.float()
            env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        remaining = clip_end - start_frames
        norm = ret / (remaining.float() * MAX_STEP_REWARD)
        counts = torch.zeros(C, device=dev).index_add_(0, clip_of, torch.ones(N, device=dev))
        norm_c = torch.zeros(C, device=dev).index_add_(0, clip_of, norm) / counts.clamp(min=1)
        fail_c = torch.zeros(C, device=dev).index_add_(0, clip_of, env.failed_eval.float()) / counts.clamp(min=1)
        sampled = (counts > 0).nonzero().flatten().tolist()
        res = {
            "norm_return": norm.mean().item(),
            "norm_return_framew": (ret.sum() / (remaining.sum() * MAX_STEP_REWARD)).item(),
            "n_failed": int(env.failed_eval.sum().item()),
            "fail_rate": env.failed_eval.float().mean().item(),
            "failed": [motion.names[i] for i in sampled if fail_c[i] > 0],
            "yaw_degrees": args.yaw_degrees,
            "xy_range_m": args.xy_range,
            "seed": args.seed,
            "n_rollouts": N,
            "root_pos_rmse_m": (root_sq_sum.sum() / steps.sum().clamp(min=1)).sqrt().item(),
            "root_pos_max_m": root_sq_max.max().sqrt().item(),
            "per_rollout_root_pos_rmse_m": (root_sq_sum / steps.clamp(min=1)).sqrt().tolist(),
            "per_rollout_root_pos_final_m": root_sq_final.sqrt().tolist(),
            "per_clip": {motion.names[i]: round(norm_c[i].item(), 4) for i in sampled},
            "per_clip_count": {motion.names[i]: int(counts[i].item()) for i in sampled},
            "per_clip_fail_rate": {motion.names[i]: round(fail_c[i].item(), 3) for i in sampled},
            "starts": [{"motion": motion.names[c], "frame": int(f - motion.clip_start[c]), "yaw": float(y), "pos_offset_m": offset}
                       for c, f, y, offset in zip(clip_of.tolist(), start_frames.tolist(), yaw.tolist(), pos_offset.tolist())],
        }
        if record:
            clip_cpu = clip_of.cpu().numpy()
            seen = np.zeros(C, dtype=np.int64)
            repeat_of = np.empty(N, dtype=np.int64)
            for i, c in enumerate(clip_cpu):
                repeat_of[i], seen[c] = seen[c], seen[c] + 1
            np.savez_compressed(record, names=np.array(motion.names), yaw=yaw.cpu().numpy(),
                                start_frame=start_frames.cpu().numpy(), init_root=env.init_root.cpu().numpy(),
                                pos_offset=pos_offset.cpu().numpy(), rollout_clip=clip_cpu, rollout_repeat=repeat_of, **{k: torch.stack(v).numpy() for k, v in traj.items()},
                                ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                                clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
        return res


    sim_cfg = make_sim(robot)
    env = TrackingSim(sim_cfg, N, robot, motion, mdp, dev, eval_mode=True, randomize=args.dr_from is not None)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        if args.record is None and (args.dr_from or args.repeats > 1):
            tag = f"random-{N}_seed-{args.seed}_yaw-{args.yaw_degrees:g}_dr-{args.dr_from or 'none'}{'' if args.dr_parts == 'all' else '-' + args.dr_parts}_x{args.repeats}"
            json.dump(res, open(args.ckpt.replace(".pt", f"_eval_{tag}.json"), "w"), indent=1)
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, "eval_random.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))[:: args.every]
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval_random.json"), "w"), indent=1)
            print(f"[eval yaw] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{N} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
