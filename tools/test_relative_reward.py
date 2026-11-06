"""CPU regression checks for episode-relative rewards and reset/replay context.

Run with: python tools/test_relative_reward.py
No motion dataset, robot assets or GPU required.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.dr import DomainRandCfg
from blinkmimic.env import TrackingSim, _randomize_reset_yaw
from blinkmimic.mdp import MDP, flatten_state
from blinkmimic.rotations import mat_to_quat, mv3, quat_to_mat, so3_exp


class RootKinematics:
    """One tracked root body, so expected tracking errors have closed forms."""
    q_lo, q_hi = torch.tensor([-2.0]), torch.tensor([2.0])
    tracked_idx = [0]

    def forward(self, p, R, v, w, q, qd):
        return p.unsqueeze(-2), R.unsqueeze(-3), v.unsqueeze(-2), w.unsqueeze(-2)


class RelativeRewardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        n = 8
        self.p = torch.randn(n, 3)
        self.R = so3_exp(torch.randn(n, 3) * 0.5)
        self.v, self.w = torch.randn(n, 3), torch.randn(n, 3)
        self.q = torch.zeros(n, 1)
        self.motion = SimpleNamespace(root_p=self.p, root_R=self.R, body_p=self.p[:, None],
                                      body_R=self.R[:, None], body_v=self.v[:, None], body_w=self.w[:, None],
                                      frame_bin=torch.arange(n), ref_state=self.ref_state)
        self.mdp = MDP(RootKinematics(), self.motion, torch.zeros(1), 1)
        self.f = torch.tensor([3, 6])
        self.f0 = torch.tensor([1, 4])
        self.a = torch.zeros(2, 1)

    def ref_state(self, f):
        return self.p[f], self.R[f], self.v[f], self.w[f], self.q[f], self.q[f]

    def transformed(self, f):
        # Independent rigid transforms of the robot episodes (including roll/pitch).
        A = so3_exp(torch.tensor([[0.3, -0.2, 0.8], [-0.2, 0.4, -0.6]]))
        t = torch.tensor([[4., -2., 1.], [-3., 1., -0.5]])
        p, R, v, w, q, qd = self.ref_state(f)
        return mv3(A, p) + t, A @ R, mv3(A, v), mv3(A, w), q, qd

    def context(self):
        S0 = self.transformed(self.f0)
        return torch.cat((S0[0], mat_to_quat(S0[1])), -1)

    def reward(self, S, f=None, **kwargs):
        return self.mdp.reward(S, self.f if f is None else f, self.a, self.a, terms=True,
                               init_root=self.context(), init_f=self.f0, **kwargs)

    def test_relative_pose_and_local_velocities(self):
        _, e = self.reward(self.transformed(self.f))
        for k in ("e_root_pos", "e_root_ori", "e_body_linvel", "e_body_angvel"):
            torch.testing.assert_close(e[k], torch.zeros(2), atol=3e-6, rtol=0)
        _, absolute = self.mdp.reward(self.transformed(self.f), self.f, self.a, self.a, terms=True)
        self.assertTrue((absolute['e_root_pos'] > 1).all())

    def test_initial_pose_and_wrong_sampled_frame(self):
        _, e = self.reward(self.transformed(self.f0), self.f0)
        torch.testing.assert_close(e['e_root_pos'], torch.zeros(2), atol=1e-10, rtol=0)
        _, wrong = self.mdp.reward(self.transformed(self.f), self.f, self.a, self.a, terms=True,
                                   init_root=self.context(), init_f=self.f0 + 1)
        self.assertTrue((wrong['e_root_pos'] > 0.01).all())

    def test_drift_penalized_and_gradients_finite(self):
        S = list(self.transformed(self.f))
        S[0] = (S[0] + torch.tensor([0.1, -0.03, 0.02])).requires_grad_()
        delta = torch.tensor([[0.04, 0.01, -0.02]]).expand(2, -1).clone().requires_grad_()
        S[1] = so3_exp(delta) @ S[1]
        S[2] = (S[2] + 0.1).requires_grad_()
        r, e = self.reward(S)
        self.assertTrue((e['e_root_pos'] > 0.01).all())
        self.assertTrue((e['e_root_ori'] > 0.001).all())
        r.sum().backward()
        for g in (S[0].grad, delta.grad, S[2].grad):
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(), 0)

    def test_compiled_batched_imagination_matches_eager(self):
        S = self.transformed(self.f)
        # Same (horizon, batch) flattening as policy_update; each sample keeps its episode context.
        expand = lambda x: x.unsqueeze(0).expand(3, *x.shape).flatten(0, 1)
        args = (tuple(expand(x) for x in S), expand(self.f), expand(self.a), expand(self.a))
        context = dict(init_root=expand(self.context()), init_f=expand(self.f0))
        eager = self.mdp.reward(*args, **context)
        compiled = torch.compile(self.mdp.reward, backend='aot_eager', fullgraph=True)(*args, **context)
        torch.testing.assert_close(compiled, eager)

    def fake_env(self):
        e = TrackingSim.__new__(TrackingSim)
        e.N, e.device, e.motion, e.mdp = 2, 'cpu', self.motion, self.mdp
        e.spec = SimpleNamespace(dr=SimpleNamespace(solref_geoms=None, com_mass_range=None))
        e.reset_noise, e.eval_mode, e.push_enabled = False, False, False
        e.pipe = SimpleNamespace(stateful=False)
        e.sim = SimpleNamespace(reset=Mock(), data=SimpleNamespace(qpos=torch.zeros(2, 8), qvel=torch.zeros(2, 7)),
                                _reset_mask=torch.zeros(2, dtype=torch.bool), _reset_mask_wp=None,
                                wp_device='cpu', use_cuda_graph=False, wp_model=None, wp_data=None)
        e.scene = SimpleNamespace(reset=Mock(), env_origins=torch.zeros(2, 3))
        ed = SimpleNamespace()
        for name in ('joint_pos_target', 'joint_vel_target', 'joint_effort_target', 'tendon_len_target',
                     'tendon_vel_target', 'tendon_effort_target', 'site_effort_target'):
            setattr(ed, name, torch.zeros(2, 1))
        e.robot = Mock(data=ed, indexing=SimpleNamespace(free_joint_q_adr=slice(0, 7), free_joint_v_adr=slice(0, 6),
                                                       joint_q_adr=[7], joint_v_adr=[6]))
        e.randomizer = Mock(_terms=[])
        for name in ('f', 'init_f', 'ep_step', 'start_bin'):
            setattr(e, name, torch.zeros(2, dtype=torch.long))
        for name in ('ep_disc_ret', 'ep_ret'):
            setattr(e, name, torch.zeros(2))
        e.init_root, e.a_prev, e.xi = torch.zeros(2, 7), torch.zeros(2, 1), torch.zeros(2, 1)
        e.obs_fresh = torch.zeros(2, dtype=torch.bool)
        return e

    def test_indexed_and_masked_reset_context(self):
        indexed, masked = self.fake_env(), self.fake_env()
        # Two partial resets must not overwrite the other episode's anchor.
        for i, f in ((0, 3), (1, 6)):
            indexed.reset(torch.tensor([i]), torch.tensor([f]))
            with patch('blinkmimic.env.wp.ScopedDevice', return_value=nullcontext()), \
                 patch('blinkmimic.env.mjwarp.reset_data'):
                masked.reset_masked(torch.arange(2) == i, torch.tensor([f, f]))
            torch.testing.assert_close(indexed.init_root, masked.init_root)
            torch.testing.assert_close(indexed.init_f, masked.init_f)
        expected = torch.cat((self.p[self.f], mat_to_quat(self.R[self.f])), -1)
        torch.testing.assert_close(indexed.init_root, expected)
        indexed.reset(torch.tensor([], dtype=torch.long))
        torch.testing.assert_close(indexed.init_root, expected)

    def test_noisy_reset_anchors_robot_after_noise_expert_before_noise(self):
        e = self.fake_env()
        e.reset_noise = True
        e.spec.dr = DomainRandCfg(reset_joint_vel=0.1)
        e.sample_frames = Mock(return_value=(self.f, torch.tensor([2, 5])))
        e.reset(torch.arange(2))
        written_pose = e.robot.write_root_link_pose_to_sim.call_args.args[0]
        torch.testing.assert_close(e.init_root, written_pose)
        torch.testing.assert_close(e.init_f, self.f)
        self.assertGreater((e.init_root[:, :3] - self.p[self.f]).abs().sum().item(), 0)
        S0 = (e.init_root[:, :3], self.R[self.f], self.v[self.f], self.w[self.f], self.q[self.f], self.q[self.f])
        _, terms = self.mdp.reward(S0, self.f, self.a, self.a, terms=True,
                                    init_root=e.init_root, init_f=e.init_f)
        torch.testing.assert_close(terms['e_root_pos'], torch.zeros(2), atol=1e-10, rtol=0)

    def test_yaw_range_tilt_and_local_velocities(self):
        n = 1024
        R = self.R[:1].expand(n, -1, -1)
        v, w = self.v[:1].expand(n, -1), self.w[:1].expand(n, -1)
        YR, Yv, Yw = _randomize_reset_yaw(R, v, w, DomainRandCfg().reset_yaw)
        Y = YR @ R.transpose(-1, -2)
        angles = torch.atan2(Y[:, 1, 0], Y[:, 0, 0])
        self.assertLessEqual(angles.abs().max().item(), torch.pi / 2)
        self.assertLess(angles.min().item(), -1.4)
        self.assertGreater(angles.max().item(), 1.4)
        torch.testing.assert_close(YR[:, 2], R[:, 2])
        torch.testing.assert_close(mv3(YR.transpose(-1, -2), Yv), mv3(R.transpose(-1, -2), v))
        torch.testing.assert_close(mv3(YR.transpose(-1, -2), Yw), mv3(R.transpose(-1, -2), w))
        for original, disabled in zip((R, v, w), _randomize_reset_yaw(R, v, w, 0)):
            self.assertIs(original, disabled)

    def test_yaw_reset_paths_and_mask_isolation(self):
        indexed, masked = self.fake_env(), self.fake_env()
        # Isolate yaw from the existing additive reset noise.
        cfg = DomainRandCfg(reset_pos=(0.,) * 3, reset_rot=(0.,) * 3,
                            reset_lin_vel=(0.,) * 3, reset_ang_vel=(0.,) * 3, reset_joint_pos=0.)
        for e in (indexed, masked):
            e.reset_noise, e.spec.dr = True, cfg
        torch.manual_seed(17)
        indexed.reset(torch.arange(2), self.f)
        torch.manual_seed(17)
        with patch('blinkmimic.env.wp.ScopedDevice', return_value=nullcontext()), \
             patch('blinkmimic.env.mjwarp.reset_data'):
            masked.reset_masked(torch.ones(2, dtype=torch.bool), self.f)
        torch.testing.assert_close(indexed.init_root, masked.init_root)
        pose = indexed.robot.write_root_link_pose_to_sim.call_args.args[0]
        vel = indexed.robot.write_root_link_velocity_to_sim.call_args.args[0]
        torch.testing.assert_close(mv3(quat_to_mat(pose[:, 3:]).transpose(-1, -2), vel[:, :3]),
                                   mv3(self.R[self.f].transpose(-1, -2), self.v[self.f]))
        torch.testing.assert_close(masked.sim.data.qvel[:, 3:6],
                                   mv3(self.R[self.f].transpose(-1, -2), self.w[self.f]))
        before = masked.init_root.clone()
        with patch('blinkmimic.env.wp.ScopedDevice', return_value=nullcontext()), \
             patch('blinkmimic.env.mjwarp.reset_data'):
            masked.reset_masked(torch.tensor([True, False]), self.f)
        torch.testing.assert_close(masked.init_root[1], before[1], atol=0, rtol=0)
        self.assertFalse(torch.equal(masked.init_root[0], before[0]))

    def test_all_tracking_rewards_invariant_to_episode_yaw(self):
        S, S0 = list(self.ref_state(self.f)), self.ref_state(self.f0)
        Y = so3_exp(torch.tensor([[0., 0., 1.5], [0., 0., -1.5]]))
        # Rotate the trajectory around each reset position, leaving the expert untouched.
        S[0] = S0[0] + mv3(Y, S[0] - S0[0])
        S[1], S[2], S[3] = Y @ S[1], mv3(Y, S[2]), mv3(Y, S[3])
        init = torch.cat((S0[0], mat_to_quat(Y @ S0[1])), -1)
        base_init = torch.cat((S0[0], mat_to_quat(S0[1])), -1)
        r, terms = self.mdp.reward(S, self.f, self.a, self.a, terms=True, init_root=init, init_f=self.f0)
        expected = self.mdp.reward(self.ref_state(self.f), self.f, self.a, self.a,
                                   init_root=base_init, init_f=self.f0)
        torch.testing.assert_close(r, expected)
        for k in ('e_root_pos', 'e_root_ori', 'e_body_pos', 'e_body_ori', 'e_body_linvel', 'e_body_angvel'):
            torch.testing.assert_close(terms[k], torch.zeros(2), atol=3e-6, rtol=0)

    def test_replay_retains_context_after_resets_and_wrap(self):
        b = ReplayBuffer(2, 2, 'cpu', 1, 1)
        tr = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
        tr.update(s=flatten_state(self.ref_state(self.f)), init_root=self.context().clone(), init_f=self.f0.clone())
        for step in range(3):
            tr['ep_step'].fill_(step)
            b.add(tr)
        expected = self.context().clone()
        tr['init_root'].zero_()  # a later reset must not mutate stored transitions
        s, _ = b.starts_at(torch.tensor([0, 1]), torch.tensor([0, 1]), 1)
        torch.testing.assert_close(s['init_root'], expected)
        torch.testing.assert_close(s['init_f'], self.f0)


if __name__ == '__main__':
    unittest.main()
