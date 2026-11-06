"""Analytic MDP heads shared by the simulator and the world model (SuperMimic paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular
velocity of the root link frame, joint positions and velocities. ``J`` is the number of joints.
"""

from __future__ import annotations

import torch

from .kinematics import Kinematics
from .motion import CONTROL_DT, MotionLib
from .reward_cfg import REWARD_CFG
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, so3_log, yaw_mat
from .sensing import SensingCfg, obs_noise


def state_dim(J: int) -> int:
    return 3 + 4 + 3 + 3 + J + J


def obs_dim(J: int) -> int:
    return J + J + 3 + 3 + J + 6


def obs_scale(J: int) -> torch.Tensor:
    # normalization of the obs before the policy (qd and angular velocity are large)
    return torch.cat(
        (torch.ones(J), torch.full((J,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(J), torch.ones(6))
    )


CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (as in tools/eval_metrics.py)
# Positive tracking upper bound; penalties and the optional contact bonus are excluded.
MAX_STEP_REWARD = CONTROL_DT * sum((
    REWARD_CFG.w_base_pos, REWARD_CFG.w_base_rotation, REWARD_CFG.w_base_lin_vel, REWARD_CFG.w_base_ang_vel,
    REWARD_CFG.w_link_pos, REWARD_CFG.w_link_rotation, REWARD_CFG.w_link_lin_vel, REWARD_CFG.w_link_ang_vel,
    REWARD_CFG.w_dofs_vel,
))


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(mm3(H.transpose(-1, -2), R)),
            mv3(Rt, v),
            mv3(Rt, w),
            q,
            qd,
        ),
        -1,
    )


def inv_dim(J: int) -> int:
    return 1 + 6 + 3 + 3 + J + J



class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, xi_dim: int,
                 sensing: SensingCfg = SensingCfg()):  # fmt: skip
        self.kin, self.motion, self.cfg = kin, motion, REWARD_CFG
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * self.cfg.soft_limit_factor, mid + half * self.cfg.soft_limit_factor
        dev = default_q.device
        J = default_q.shape[0]
        self.num_j = J
        self.obs_dim, self.xi_dim = obs_dim(J), xi_dim
        self.critic_dim = 3 * J + len(kin.tracked_idx) * 12 + 12 + self.xi_dim
        self.obs_noise = obs_noise(sensing, J).to(dev)
        self.obs_scale = obs_scale(J).to(dev)
        # InstantTrack contact reward (training only): tracked-body indices of the feet, set by the trainer
        self.feet: list[int] = []

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + q.shape[-1]]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(mm3(Rt, self.motion.root_R[f]))
        return torch.cat((q - self.default_q - dq0, qd, mv3(Rt, w), g_b, a_prev, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def base_diff(self, S, f, init_root=None, init_f=None):
        """Reference base diffs: initial-frame displacement/rotation, current-base local velocities.

        ``init_root`` is robot xyz+wxyz after reset noise; ``init_f`` selects the expert frame before
        noise. Without context, position/orientation compare absolute poses for standalone playback.
        Rotation diff is 2 * sign(q_err.w) * q_err.xyz, not a rotation vector/geodesic angle.
        Velocity diffs use each current base frame; initial velocities are not subtracted.
        """
        assert (init_root is None) == (init_f is None), "reset pose and expert frame must be supplied together"
        p, R, v, w = S[:4]
        m = self.motion
        rp, rR = m.root_p[f], m.root_R[f]
        if init_root is None:
            diff_pos, rel_R, ref_rel_R = rp - p, R, rR
        else:
            p0, R0 = init_root[..., :3], quat_to_mat(init_root[..., 3:7])
            rp0, rR0 = m.root_p[init_f], m.root_R[init_f]
            R0t, rR0t = R0.transpose(-1, -2), rR0.transpose(-1, -2)
            diff_pos = mv3(rR0t, rp - rp0) - mv3(R0t, p - p0)
            rel_R, ref_rel_R = mm3(R0t, R), mm3(rR0t, rR)
        q_err = mat_to_quat(mm3(rel_R.transpose(-1, -2), ref_rel_R))
        sign = torch.where(q_err[..., :1] < 0, -1.0, 1.0)
        diff_rot = 2 * sign * q_err[..., 1:]
        Rt, rRt = R.transpose(-1, -2), rR.transpose(-1, -2)
        diff_lv = mv3(rRt, m.root_v[f]) - mv3(Rt, v)
        diff_av = mv3(rRt, m.root_w[f]) - mv3(Rt, w)
        return diff_pos, diff_rot, diff_lv, diff_av

    def hybrid_root(self, R, f):
        """Robot's current yaw with the reference's current roll/pitch; not a reset-frame transform."""
        rR = self.motion.root_R[f]

        def heading(A):
            x, y = A[..., 0, 0], A[..., 1, 0]
            valid = x.square() + y.square() > 1e-12
            # At a vertical forward axis yaw is undefined: use zero, keeping a valid rotation and finite gradients.
            return torch.atan2(torch.where(valid, y, 0.0), torch.where(valid, x, 1.0))

        delta = heading(R) - heading(rR)
        c, s = delta.cos(), delta.sin()
        z, one = torch.zeros_like(c), torch.ones_like(c)
        Y = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(R.shape)
        return mm3(Y, rR)

    def body_diff(self, S, f, body_state=None):
        """Link diffs in reference-root / hybrid-root frames, with rotation as a shortest-path 3D rotvec.

        Subtract each own root for positions. Velocities are full link world velocities rotated into
        those frames; do not subtract base velocity. ``body_state`` reuses FK when reward needs contacts.
        All four outputs have shape (..., tracked_links, 3).
        """
        p, R = S[:2]
        bp, bR, bv, bw = self.kin.forward(*S) if body_state is None else body_state
        m = self.motion
        rp, rR = m.root_p[f], m.root_R[f]
        rRt = rR.transpose(-1, -2).unsqueeze(-3)
        Ht = self.hybrid_root(R, f).transpose(-1, -2).unsqueeze(-3)
        diff_pos = mv3(rRt, m.body_p[f] - rp.unsqueeze(-2)) - mv3(Ht, bp - p.unsqueeze(-2))
        ref_local_R, local_R = mm3(rRt, m.body_R[f]), mm3(Ht, bR)
        diff_rot = so3_log(mm3(local_R.transpose(-1, -2), ref_local_R))
        diff_lv = mv3(rRt, m.body_v[f]) - mv3(Ht, bv)
        diff_av = mv3(rRt, m.body_w[f]) - mv3(Ht, bw)
        return diff_pos, diff_rot, diff_lv, diff_av

    def reward(self, S, f, a, a_prev, terms: bool = False, w_contact: float = 0.0,
               init_root=None, init_f=None):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f.

        ``init_root`` is the robot's reset pose (xyz + wxyz), after reset noise; ``init_f`` is the
        sampled expert frame, before noise. Root tracking compares displacement and rotation in each
        initial root frame. Both must be supplied together; omitted context retains absolute tracking
        for standalone reference playback. Context is per sample, including replay/imagination batches.

        ``w_contact`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``w_contact = 0``."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        body_state = self.kin.forward(p, R, v, w, q, qd)
        bp, bR, bv, bw = body_state
        diff_pos, diff_rot, diff_lv, diff_av = self.body_diff(S, f, body_state)
        # Diagnostic errors are mean squared vector norms; rewards below use the configured sigma aggregations.
        e_pos = (diff_pos ** 2).sum(-1).mean(-1)
        e_ori = (diff_rot ** 2).sum(-1).mean(-1)
        e_lv = (diff_lv ** 2).sum(-1).mean(-1)
        e_av = (diff_av ** 2).sum(-1).mean(-1)
        diff_rp, diff_ro, diff_rlv, diff_rav = self.base_diff(S, f, init_root, init_f)
        e_rp = (diff_rp ** 2).sum(-1)
        e_ro = (diff_ro ** 2).sum(-1)
        e_rlv = (diff_rlv ** 2).sum(-1)
        e_rav = (diff_rav ** 2).sum(-1)
        e_jv = ((m.qd[f] - qd) ** 2).sum(-1)
        n_links = diff_pos.shape[-2]
        # Keep diagnostic errors as mean per-link vector norms. Reward sigma follows the reference
        # directly: pos/rot sum all link components; velocities average all components (including xyz).
        r_track = (
            c.w_link_pos * torch.exp(-e_pos * n_links / c.sigma_link_pos)
            + c.w_link_rotation * torch.exp(-e_ori * n_links / c.sigma_link_rotation)
            + c.w_link_lin_vel * torch.exp(-e_lv / (3 * c.sigma_link_lin_vel))
            + c.w_link_ang_vel * torch.exp(-e_av / (3 * c.sigma_link_ang_vel))
            + c.w_base_pos * torch.exp(-e_rp / c.sigma_base_pos)
            + c.w_base_rotation * torch.exp(-e_ro / c.sigma_base_rotation)
            + c.w_base_lin_vel * torch.exp(-e_rlv / c.sigma_base_lin_vel)
            + c.w_base_ang_vel * torch.exp(-e_rav / c.sigma_base_ang_vel)
            + c.w_dofs_vel * torch.exp(-e_jv / c.sigma_dofs_vel)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_mag = (a ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_action_magnitude * r_mag + c.w_joint_limit * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz, rz = bp[..., self.feet, 2], m.body_p[f][..., self.feet, 2]
            c_sim = torch.sigmoid((CONTACT_Z - fz) / 0.005)
            rc = getattr(m, "ref_contact", None)  # MotionLib.set_ref_contact (--ref_contact_fix), else the fixed 4.5 cm
            c_ref = rc[f] if rc is not None else (rz < CONTACT_Z).float()
            slip = (bv[..., self.feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "e_root_ori": e_ro,
                "e_root_linvel": e_rlv,
                "e_root_angvel": e_rav,
                "e_body_linvel": e_lv,
                "e_body_angvel": e_av,
                "e_joint_vel": e_jv,
                "action_rate": r_rate,
                "action_magnitude": r_mag,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi, init_root, init_f) -> torch.Tensor:
        """Joints, body diffs, the reward's four base diffs, and sim context (goal appended by trainer).

        Base height, tilt, raw local velocities and absolute root pose errors are replaced by the
        same 12 base-diff components used by reward. Body inputs are the four shared link diffs.
        Reset context is required for every sample.
        """
        assert init_root is not None and init_f is not None, "critic needs each sample's reset pose and expert frame"
        q, qd = S[4:]
        body = tuple(x.flatten(-2) for x in self.body_diff(S, f))
        base = self.base_diff(S, f, init_root, init_f)
        return torch.cat((q, qd, self.motion.q[f] - q, *body, *base, xi), -1)
