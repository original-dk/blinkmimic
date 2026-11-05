"""Motion-tracking simulator on Genesis.

The environment is a thin batched wrapper: it applies the joint setpoint, steps Genesis, labels
the transition with the analytic reward/termination of :mod:`blinkmimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (SuperMimic paper Appendix D.1). Domain randomization
(:mod:`blinkmimic.dr`) and the policy's sensing (:mod:`blinkmimic.sensing`) follow ``RobotSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
import torch

from .dr import DomainRandomizer, XiLayout
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline

DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION


@dataclass(frozen=True)
class SimulationCfg:
    dt: float = PHYSICS_DT
    iterations: int = 100


def make_sim(robot: RobotSpec) -> SimulationCfg:
    return SimulationCfg()


def _bin_ema_w(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, wt: torch.Tensor, old: float) -> torch.Tensor:
    """_bin_ema over the samples with weight wt = 1 (the others weight 0): fixed size, no host-side selection."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


def _bin_ema(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, old: float) -> torch.Tensor:
    """EMA update of per-bin statistics with possibly repeated bin indices: the samples of a bin are averaged, then
    one EMA step is taken for every bin that received at least one sample (no host sync)."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x)
    cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


class TrackingSim:
    """Batched motion-tracking simulator for the robot of ``robot``."""

    def __init__(self, sim_cfg: SimulationCfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP,
                 device: str, gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False):  # fmt: skip
        """``randomize`` (default: not eval_mode) switches domain randomization, observation noise and
        delay, and pushes; ``eval_mode`` switches evaluation bookkeeping (no auto-reset, episode starts
        exactly on the reference)."""
        self.device = device
        if not gs._initialized:
            gs.init(backend=gs.cpu if torch.device(device).type == "cpu" else gs.cuda, logging_level="warning")
        requested = torch.device(device)
        if requested.type != gs.device.type or (requested.index is not None and requested.index != gs.device.index):
            raise ValueError(f"Genesis is initialized on {gs.device}, requested {device}")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim_cfg.dt),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, iterations=sim_cfg.iterations,
                                                 batch_links_info=True, batch_dofs_info=True),
            show_viewer=False,
        )
        self.ground = self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=robot.dr.friction_range[0]))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot.mjcf_path, align=False),
                                           material=gs.materials.Rigid(friction=1.0))
        self.scene.build(n_envs=num_envs, env_spacing=(0.0, 0.0))
        self.dof_ids = [self.robot.get_joint(n).dofs_idx_local[0] for n in robot.joint_names]
        self.q_ids = [self.robot.get_joint(n).qs_idx_local[0] for n in robot.joint_names]
        root = next(j for j in self.robot.joints if j.type == gs.JOINT_TYPE.FREE)
        self.root_dofs, self.root_qs = root.dofs_idx_local, root.qs_idx_local
        self.root_link = self.robot.get_link(robot.tracked_bodies[0]).idx_local
        t = lambda values: torch.tensor(values, device=device, dtype=torch.float32)  # noqa: E731
        self.kp = t([d.stiffness for d in robot.drives])
        self.kd = t([d.damping for d in robot.drives])
        self.armature = t([d.armature for d in robot.drives])
        effort = t([d.effort_limit for d in robot.drives])
        self.robot.set_dofs_kp(self.kp, self.dof_ids)
        self.robot.set_dofs_kv(self.kd, self.dof_ids)
        self.robot.set_dofs_armature(self.armature, self.dof_ids)
        self.robot.set_dofs_force_range(-effort, effort, self.dof_ids)
        self.N = num_envs
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = torch.tensor(robot.default_joint_pos_list(), device=dev)
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)

        # ---- sim context xi (layout in blinkmimic.dr; paper setup: friction, joint offsets, com offset) ----
        self.layout = XiLayout(robot, dev)
        assert self.layout.dim == mdp.xi_dim, (self.layout.dim, mdp.xi_dim)
        self.xi = torch.zeros(self.N, self.layout.dim, device=dev)
        self.randomizer = DomainRandomizer(robot, self.layout, self, eval_mode=not self.randomize)
        self.randomizer.startup(self.xi)

        # ---- sensing (stateful signal chain only if the robot defines one) ----
        self.pipe = ObsPipeline(robot, mdp, self.layout, dev, PHYSICS_DT)
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=dev)
        self._obs = self._obs_state = None
        self.push_timer = torch.zeros(self.N, device=dev)
        self.push_enabled = robot.dr.push_interval_s is not None and self.randomize
        if self.push_enabled:
            self.push_timer.uniform_(*robot.dr.push_interval_s)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, self.J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        # InstantTrack: share of resets sent one bin before recent failure points (per-bin termination rate)
        self.fail_reset = fail_reset
        # index_put_ with duplicate bin indices keeps one arbitrary episode per bin and step (undefined
        # behaviour); bin_fix averages the episodes of a bin first (as supermimic_vibed's update_bin_returns)
        self.bin_fix = bin_fix
        # --eprobe_share: externally set reset distribution (train.py: the windows before the deployed
        # policy's evaluation-condition failures), mixed in with share focus_share
        self.focus_q, self.focus_share = None, 0.0
        self.focus_static, self.focus_share_t = False, None  # use_static_focus()
        self.bin_fail = torch.zeros(motion.num_bins, device=dev)
        prev = torch.arange(motion.num_bins, device=dev) - 1
        self.bin_prev = torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    # ------------------------------------------------------------------ state
    def state(self):
        p = self.robot.get_links_pos([self.root_link])[:, 0]
        R = quat_to_mat(self.robot.get_links_quat([self.root_link])[:, 0])
        v = self.robot.get_links_vel([self.root_link])[:, 0]
        w = self.robot.get_links_ang([self.root_link])[:, 0]
        return (p, R, v, w, self.robot.get_dofs_position(self.dof_ids), self.robot.get_dofs_velocity(self.dof_ids))

    def _write_state(self, env_ids, p, R, v, w, q, qd):
        # Free-joint linear velocity is world-frame; angular velocity is root-local.
        self.robot.set_qpos(torch.cat((p, mat_to_quat(R)), -1), self.root_qs, env_ids, zero_velocity=False)
        self.robot.set_qpos(q, self.q_ids, env_ids, zero_velocity=False)
        w_local = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        self.robot.set_dofs_velocity(torch.cat((v, w_local), -1), self.root_dofs, env_ids)
        self.robot.set_dofs_velocity(qd, self.dof_ids, env_ids)
        self.robot.control_dofs_position(q, self.dof_ids, env_ids)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int, gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``gate`` (reset_masked): a 0-dim bool; the running average bin_w_avg only advances where it is true (the
        eager reset samples only when an episode ended)."""
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        if gate is None:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        else:
            self.bin_w_avg.copy_(torch.where(gate, self.bin_w_avg * 0.999 + 0.001 * soft, self.bin_w_avg))
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        if self.focus_static:  # reset_masked: device-side focus (share 0 leaves probs bitwise unchanged)
            probs = (1 - self.focus_share_t) * probs + self.focus_share_t * self.focus_q
        elif self.focus_q is not None and self.focus_share > 0:
            probs = (1 - self.focus_share) * probs + self.focus_share * self.focus_q
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m, c = self.motion, self.spec.dr
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self._write_state(env_ids, p, R, v, w, q, qd)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.randomizer.reset(self.xi, env_ids)
        self.obs_fresh.index_fill_(0, env_ids, True)

    def use_static_focus(self):
        """Keep the focus distribution and share in fixed device tensors, updated in place by set_focus."""
        self.focus_static = True
        self.focus_q = torch.full((self.motion.num_bins,), 1.0 / self.motion.num_bins, device=self.device)
        self.focus_share_t = torch.zeros((), device=self.device)

    def set_focus(self, q: torch.Tensor, share: float):
        if self.focus_static:
            self.focus_q.copy_(q), self.focus_share_t.fill_(share)
        else:
            self.focus_q, self.focus_share = q, share

    def reset_masked(self, mask: torch.Tensor, frames: torch.Tensor | None = None):
        """Reset selected worlds. Genesis setters use explicit environment indices."""
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        self.reset(ids, None if frames is None else frames[ids])

    def step_tail_masked(self, out: dict) -> dict:
        """step_tail() with masked bin-statistic updates and indexed Genesis resets. Instead
        of per-episode tensors, out gets n_done and the sums ep_len_sum / ep_ret_norm_sum over the ended episodes."""
        assert self.bin_fix, "reset_masked needs --bin_fix (the index_put_ update keeps an arbitrary episode per bin)"
        done, term, f = out["done"], out["term"], out["f"]
        dw = done.float()
        T = self.ep_step.float()
        boot = torch.where(term, 0.0, (self.gamma**T) * (self.ep_ret / T) / (1 - self.gamma))
        G = self.ep_disc_ret + boot
        self.bin_G.copy_(_bin_ema_w(self.bin_G, self.start_bin, G, dw, 0.8))
        if self.fail_reset > 0:
            fb = self.motion.frame_bin[f.clamp(max=self.motion.num_frames - 1)]
            bf = torch.where(done.any(), self.bin_fail * 0.999, self.bin_fail)
            self.bin_fail.copy_(_bin_ema_w(bf, fb, term.float(), dw, 0.9))
        out["n_done"] = dw.sum()
        out["ep_len_sum"] = (T * dw).sum()
        out["ep_ret_norm_sum"] = (self.ep_ret / (T * MAX_STEP_REWARD) * dw).sum()
        self.reset_masked(done)
        return out

    def refresh(self):
        """Genesis state setters update forward kinematics immediately."""

    # ------------------------------------------------------------------ observation
    def _imu_site(self) -> tuple[torch.Tensor, torch.Tensor]:
        S = self.state()
        Rs = S[1] @ self.pipe.site_rot
        gyro = (Rs.transpose(-1, -2) @ S[3].unsqueeze(-1)).squeeze(-1)
        acc = self.robot.get_links_acc([self.root_link])[:, 0]
        ang_acc = self.robot.get_links_acc_ang([self.root_link])[:, 0]
        offset = (S[1] @ self.pipe.site_pos.expand(self.N, -1).unsqueeze(-1)).squeeze(-1)
        acc = acc + torch.cross(ang_acc, offset, dim=-1) + torch.cross(S[3], torch.cross(S[3], offset, dim=-1), dim=-1)
        acc = acc + acc.new_tensor((0.0, 0.0, 9.81))
        return gyro, (Rs.transpose(-1, -2) @ acc.unsqueeze(-1)).squeeze(-1)

    def observe(self) -> torch.Tensor:
        """Policy observation of the current state (training: noisy; eval: noise- and delay-free)."""
        S = self.state()
        if not self.pipe.stateful:
            o = self.mdp.obs(S, self.a_prev, self.xi, self.f)
            self._obs = self.mdp.add_noise(o) if self.randomize else o
            return self._obs
        if self._obs_state is None or self.obs_fresh.any():
            # just reset: filters start converged on the true values (accelerometer: gravity only)
            fresh = self.obs_fresh.unsqueeze(-1)
            Rs = S[1] @ self.pipe.site_rot if self.spec.sensing.imu is not None else S[1]
            gyro = (Rs.transpose(-1, -2) @ S[3].unsqueeze(-1)).squeeze(-1)
            accel = Rs[..., 2, :] * 9.81
            init = self.pipe.init_state(S, self.xi, gyro, accel)
            init["prev_raw"] = self.pipe.raw_obs(init, S, self.a_prev, self.xi, self.f)
            st = init if self._obs_state is None else {k: torch.where(fresh, init[k], x) for k, x in self._obs_state.items()}
        else:
            st = self._obs_state
        raw = self.pipe.raw_obs(st, S, self.a_prev, self.xi, self.f)
        self._obs, self._obs_state = self.pipe.observe(st, raw, noise=self.randomize, delay=self.randomize)
        self.obs_fresh[:] = False
        return self._obs

    def _push(self, done: torch.Tensor) -> torch.Tensor:
        """Root velocity kicks at random intervals (between steps, after s1 is recorded)."""
        self.push_timer -= CONTROL_DT
        due = (self.push_timer <= 0) & ~done
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            c = self.spec.dr
            _, R, v, w, _, _ = self.state()
            kick = torch.stack([torch.empty(len(ids), device=self.device).uniform_(-h, h) for h in c.push_velocity], -1)
            vel = torch.cat((v[ids], w[ids]), -1) + kick
            vel[:, 3:] = (R[ids].transpose(-1, -2) @ vel[:, 3:].unsqueeze(-1)).squeeze(-1)
            self.robot.set_dofs_velocity(vel, self.root_dofs, ids)
            self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*c.push_interval_s)
        return due

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        out = self.step_core(a)
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
            return out
        return self.step_tail(out)

    def step_core(self, a: torch.Tensor) -> dict:
        """The step up to the episode ends, using eager Genesis stepping.
        The simulator wrapper rebinds f, ep_step and a_prev."""
        if self.pipe.stateful and self._obs is None:
            self.observe()
        o, o_state = self._obs, self._obs_state
        self._obs = None
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1 : 1 + self.J] + self.alpha * a
        self.robot.control_dofs_position(target, self.dof_ids)
        for _ in range(DECIMATION):
            self.scene.step()
            if self.pipe.stateful:
                gyro, accel = self._imu_site() if self.spec.sensing.imu is not None else (None, None)
                self._obs_state = self.pipe.filter_update(self._obs_state, self.robot.get_dofs_velocity(self.dof_ids), gyro, accel, self.xi,
                                                          PHYSICS_DT, noise=self.randomize)  # fmt: skip
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "cut": done,
        }  # fmt: skip
        if self.pipe.stateful:
            out["o"], out["obs_state"] = o, self.pipe.pack(o_state)
        if self.push_enabled:
            out["cut"] = done | self._push(done)
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        return out

    def step_tail(self, out: dict) -> dict:
        """Episode ends of a training step: bin statistics and resets of the ended episodes (host-side ids)."""
        done, term, f = out["done"], out["term"], out["f"]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            if self.bin_fix:
                self.bin_G.copy_(_bin_ema(self.bin_G, b, G, 0.8))
            else:
                self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            if self.fail_reset > 0:
                # termination rate of the bin where each episode ended (decays toward 0 as bins are mastered)
                fb = self.motion.frame_bin[f[ids].clamp(max=self.motion.num_frames - 1)]
                self.bin_fail.mul_(0.999)  # bins no episode ends in any more fade out
                if self.bin_fix:
                    self.bin_fail.copy_(_bin_ema(self.bin_fail, fb, term[ids].float(), 0.9))
                else:
                    self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * term[ids].float())
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
