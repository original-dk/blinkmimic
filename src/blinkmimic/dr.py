"""Simulator domain randomization and the layout of the sim context xi.

Every randomized simulator parameter is part of xi, which the world model and critic see (the
policy does not). The defaults are the paper's G1 setup: friction once per env, joint (encoder)
offsets per episode, and a CoM offset of ``RobotSpec.com_body`` once per env. A robot can switch on
more terms in ``RobotSpec.dr``; Genesis setters apply them to the selected worlds, and xi records
the applied values. Per-environment contact solref is not supported by Genesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
    from .robot import RobotSpec


@dataclass(frozen=True)
class DomainRandCfg:
    friction_range: tuple[float, float] = (0.3, 1.2)
    """Contact friction, sampled once per env."""
    friction_geoms: str | None = None
    """Regex of the robot geoms that get it (None: every robot geom). The ground uses the bottom of the range; Genesis takes the maximum contact friction of the pair."""
    joint_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    """Joint regex -> half-range (rad) of the encoder offset, per episode: the policy reads q + bias and
    the PD target moves by -bias, as on a robot whose encoder zero is off."""
    com_mass_range: tuple[float, float] | None = None
    """None: ``com_body`` CoM offset once per env. Otherwise ``com_body`` is randomized per episode by
    mass and inertia scale in this range and CoM shift in ``RobotSpec.com_range``."""
    link_mass_range: tuple[float, float] | None = None
    """Per-link mass scale (inertia scaled with it), per episode."""
    link_mass_exclude: tuple[str, ...] = ()
    """Bodies kept out of the link-mass term (``com_body`` always is)."""
    solref_geoms: str | None = None
    solref_timeconst: tuple[float, float] = (0.02, 0.02)
    solref_dampratio: tuple[float, float] = (1.0, 1.0)
    """Contact solref of ``solref_geoms`` (shared across them), per episode."""
    pd_gain_scale: tuple[tuple[tuple[int, ...], tuple[float, float], tuple[float, float]], ...] = ()
    """(joint ids in joint_names order, kp scale range, kd scale range), per joint and episode."""
    armature_range: tuple[float, float] | None = None
    damping_range: tuple[float, float] | None = None
    """Per-joint scale of the rotor armature / passive joint damping, per episode."""
    push_interval_s: tuple[float, float] | None = None
    push_velocity: tuple[float, ...] = (0.0,) * 6
    """Root velocity kick (x, y, z, roll, pitch, yaw half-ranges) every push_interval_s."""
    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    reset_joint_vel: float = 0.0
    """Half-ranges of the offsets from the reference state at episode start."""


def _match(patterns: dict[str, float], names, default: float = 0.0) -> list[float]:
    out = []
    for n in names:
        val = default
        for pat, v in patterns.items():
            if re.fullmatch(pat, n):
                val = v
        out.append(val)
    return out


class XiLayout:
    """Slices of xi and its normalization (center, half-range) for the networks."""

    def __init__(self, robot: RobotSpec, device: str):
        c, s = robot.dr, robot.sensing
        J = robot.num_joints
        self.slices: dict[str, slice] = {}
        centers, scales = [], []

        def add(name: str, center: list[float], scale: list[float]):
            n = sum(sl.stop - sl.start for sl in self.slices.values())
            self.slices[name] = slice(n, n + len(center))
            centers.extend(center), scales.extend(scale)

        lo, hi = c.friction_range
        add("friction", [(lo + hi) / 2], [(hi - lo) / 2])
        add("joint_offset", [0.0] * J, [max(h, 1e-6) for h in _match(c.joint_offset, robot.joint_names)])
        add("com", [0.0] * 3, [0.05] * 3)  # fixed scale (the paper setup's normalization)
        if c.com_mass_range is not None:
            add("com_mass", [sum(c.com_mass_range) / 2], [(c.com_mass_range[1] - c.com_mass_range[0]) / 2])
        self.link_bodies: list[str] = []
        if c.link_mass_range is not None:
            m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
            excl = (robot.com_body, *c.link_mass_exclude)
            self.link_bodies = [m.body(b).name for b in range(1, m.nbody)
                                if m.body_mass[b] > 0 and not any(re.fullmatch(p, m.body(b).name) for p in excl)]  # fmt: skip
            add("link_mass", [sum(c.link_mass_range) / 2] * len(self.link_bodies),
                [(c.link_mass_range[1] - c.link_mass_range[0]) / 2] * len(self.link_bodies))  # fmt: skip
        if c.solref_geoms is not None:
            add("solref", [sum(c.solref_timeconst) / 2, sum(c.solref_dampratio) / 2],
                [max((c.solref_timeconst[1] - c.solref_timeconst[0]) / 2, 1e-6),
                 max((c.solref_dampratio[1] - c.solref_dampratio[0]) / 2, 1e-6)])  # fmt: skip
        if c.pd_gain_scale:
            add("kp", [1.0] * J, [0.2] * J)
            add("kd", [1.0] * J, [0.5] * J)
        if c.armature_range is not None:
            add("armature", [sum(c.armature_range) / 2] * J, [(c.armature_range[1] - c.armature_range[0]) / 2] * J)
        if c.damping_range is not None:
            add("damping", [sum(c.damping_range) / 2] * J, [(c.damping_range[1] - c.damping_range[0]) / 2] * J)
        if s.imu is not None:
            add("imu_tilt", [0.0, 0.0], [max(s.imu.tilt, 1e-6)] * 2)
            add("gyro_bias", [0.0] * 3, [max(s.imu.gyro_bias, 1e-6)] * 3)
        if s.joint_vel_filter is not None:
            lo, hi = s.joint_vel_filter.tau_scale_range
            add("qd_tau_scale", [(lo + hi) / 2], [max((hi - lo) / 2, 1e-6)])
        self.dim = len(centers)
        self.center = torch.tensor(centers, device=device)
        self.scale = torch.tensor(scales, device=device)

    def normalize(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.center) / self.scale

    def get(self, xi: torch.Tensor, name: str) -> torch.Tensor:
        return xi[..., self.slices[name]]


class DomainRandomizer:
    """Apply Genesis per-world parameters and record them in the world-model context xi."""

    def __init__(self, robot: RobotSpec, layout: XiLayout, env, eval_mode: bool):
        self.cfg, self.spec, self.layout, self.env = robot.dr, robot, layout, env
        self.eval_mode = eval_mode
        c, ent, dev = self.cfg, env.robot, env.device
        if c.solref_geoms is not None:
            raise ValueError("Genesis does not support per-environment geom solref randomization")
        self.J = robot.num_joints
        self.com_body_id = ent.get_link(robot.com_body).idx_local
        self.joint_offset_half = torch.tensor(_match(c.joint_offset, robot.joint_names), device=dev)
        self.default_com = ent.get_links_COM().clone()
        self.default_mass = ent.get_links_mass().clone()
        self.default_inertia = ent.get_links_inertia().clone()
        self.default_damping = ent.get_dofs_damping(env.dof_ids).clone()
        self.link_body_ids = [ent.get_link(n).idx_local for n in layout.link_bodies]
        self.friction_links = list(range(ent.n_links))
        if c.friction_geoms is not None:
            m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
            selected, other = set(), set()
            for g in range(m.ngeom):
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                b = ent.get_link(m.body(int(m.geom_bodyid[g])).name).idx_local
                (selected if re.fullmatch(c.friction_geoms, m.geom(g).name or "") else other).add(b)
            if selected & other:
                raise ValueError("Genesis friction selection must include all collision geoms of each selected link")
            if not selected:
                raise ValueError(f"No collision geoms match {c.friction_geoms!r}")
            self.friction_links = sorted(selected)

    def _uniform(self, n, width, bounds):
        return torch.empty(n, width, device=self.env.device).uniform_(*bounds)

    # ------------------------------------------------------------------ once per env
    def startup(self, xi: torch.Tensor):
        c, ent, N, L = self.cfg, self.env.robot, self.env.num_envs, self.layout
        friction = torch.ones(N, 1, device=xi.device) if self.eval_mode else self._uniform(N, 1, c.friction_range)
        ent.set_friction_ratio(friction.expand(-1, len(self.friction_links)), self.friction_links)
        xi[:, L.slices["friction"]] = friction
        if c.com_mass_range is None and not self.eval_mode:
            com = torch.cat([self._uniform(N, 1, bounds) for bounds in self.spec.com_range], -1)
            b = self.com_body_id
            ent.set_links_COM(self.default_com[:, b:b + 1] + com[:, None], [b])
            xi[:, L.slices["com"]] = ent.get_links_COM([b])[:, 0] - self.default_com[:, b]
        for name in ("com_mass", "link_mass", "kp", "kd", "armature", "damping", "qd_tau_scale"):
            if name in L.slices:
                xi[:, L.slices[name]] = 1.0

    # ------------------------------------------------------------------ per episode
    def reset_masked(self, xi: torch.Tensor, mask: torch.Tensor):
        self.reset(xi, mask.nonzero(as_tuple=False).squeeze(-1))

    def reset(self, xi: torch.Tensor, env_ids: torch.Tensor):
        if self.eval_mode or len(env_ids) == 0:
            return
        c, ent, env, L = self.cfg, self.env.robot, self.env, self.layout
        n, dev = len(env_ids), env.device
        xi[env_ids, L.slices["joint_offset"]] = (2 * torch.rand(n, self.J, device=dev) - 1) * self.joint_offset_half
        for name, bodies, bounds in (("com_mass", [self.com_body_id], c.com_mass_range),
                                     ("link_mass", self.link_body_ids, c.link_mass_range)):
            if bounds is None:
                continue
            scale = self._uniform(n, len(bodies), bounds)
            mass = self.default_mass[env_ids][:, bodies]
            inertia = self.default_inertia[env_ids][:, bodies]
            ent.set_links_mass(mass * scale, bodies, env_ids)
            ent.set_links_inertia(inertia * scale[..., None, None], bodies, env_ids)
            xi[env_ids, L.slices[name]] = ent.get_links_mass(bodies, env_ids) / mass
        if c.com_mass_range is not None:
            b = self.com_body_id
            com = torch.cat([self._uniform(n, 1, bounds) for bounds in self.spec.com_range], -1)
            ent.set_links_COM(self.default_com[env_ids, b:b + 1] + com[:, None], [b], env_ids)
            xi[env_ids, L.slices["com"]] = ent.get_links_COM([b], env_ids)[:, 0] - self.default_com[env_ids, b]
        if c.pd_gain_scale:
            kp, kd = torch.ones(n, self.J, device=dev), torch.ones(n, self.J, device=dev)
            for ids, kp_range, kd_range in c.pd_gain_scale:
                kp[:, ids] = self._uniform(n, len(ids), kp_range)
                kd[:, ids] = self._uniform(n, len(ids), kd_range)
            ent.set_dofs_kp(kp * env.kp, env.dof_ids, env_ids)
            ent.set_dofs_kv(kd * env.kd, env.dof_ids, env_ids)
            xi[env_ids, L.slices["kp"]] = ent.get_dofs_kp(env.dof_ids, env_ids) / env.kp
            xi[env_ids, L.slices["kd"]] = ent.get_dofs_kv(env.dof_ids, env_ids) / env.kd
        if c.armature_range is not None:
            scale = self._uniform(n, self.J, c.armature_range)
            ent.set_dofs_armature(scale * env.armature, env.dof_ids, env_ids)
            xi[env_ids, L.slices["armature"]] = ent.get_dofs_armature(env.dof_ids, env_ids) / env.armature
        if c.damping_range is not None:
            scale = self._uniform(n, self.J, c.damping_range)
            default = self.default_damping[env_ids]
            ent.set_dofs_damping(scale * default, env.dof_ids, env_ids)
            applied = ent.get_dofs_damping(env.dof_ids, env_ids)
            xi[env_ids, L.slices["damping"]] = torch.where(default > 0, applied / default.clamp(min=1e-9), 1.0)
        s = self.spec.sensing
        if s.imu is not None:
            xi[env_ids, L.slices["imu_tilt"]] = self._uniform(n, 2, (-s.imu.tilt, s.imu.tilt))
            xi[env_ids, L.slices["gyro_bias"]] = self._uniform(n, 3, (-s.imu.gyro_bias, s.imu.gyro_bias))
        if s.joint_vel_filter is not None:
            xi[env_ids, L.slices["qd_tau_scale"]] = self._uniform(n, 1, s.joint_vel_filter.tau_scale_range)
