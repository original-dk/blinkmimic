"""Base-only comparison against the supplied reference quaternion and local-velocity equations.

Run: python tools/test_base_reference.py
"""

import math
import unittest

import torch

from blinkmimic.mdp import MAX_STEP_REWARD
from blinkmimic.motion import CONTROL_DT
from blinkmimic.rotations import mat_to_quat, mv3, so3_exp
import test_relative_reward as relative


def mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def inv(q):
    return q * q.new_tensor([1., -1., -1., -1.])


class BaseReferenceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = relative.RelativeRewardTest()
        self.fixture.setUp()
        self.mdp = self.fixture.mdp

    def test_diff_matches_reference_quaternion_algebra(self):
        x = self.fixture
        S = list(x.transformed(x.f))
        S[0] = S[0] + torch.tensor([.2, -.4, .1])
        S[1] = so3_exp(torch.tensor([[.6, -.3, .8], [-.4, .2, -.9]])) @ S[1]
        init = x.context()
        dp, dr, dv, dw = self.mdp.base_diff(S, x.f, init, x.f0)
        q = mat_to_quat(S[1])
        qm, qm0 = mat_to_quat(x.R[x.f]), mat_to_quat(x.R[x.f0])
        q_rel = mul(inv(init[:, 3:]), q)
        qm_rel = mul(inv(qm0), qm)
        err = mul(inv(q_rel), qm_rel)
        sign = torch.where(err[:, :1] < 0, -1., 1.)
        torch.testing.assert_close(dr, 2 * sign * err[:, 1:], atol=1e-6, rtol=1e-5)
        # Independent quaternion-vector rotation, as transform_by_quat in the reference.
        def rotate(q, v):
            pure = torch.cat((torch.zeros_like(v[:, :1]), v), -1)
            return mul(mul(q, pure), inv(q))[:, 1:]
        expected_p = rotate(inv(qm0), x.p[x.f] - x.p[x.f0]) - rotate(inv(init[:, 3:]), S[0] - init[:, :3])
        torch.testing.assert_close(dp, expected_p, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(dv, rotate(inv(qm), x.v[x.f]) - rotate(inv(q), S[2]))
        torch.testing.assert_close(dw, rotate(inv(qm), x.w[x.f]) - rotate(inv(q), S[3]))

    def test_rotation_uses_quaternion_chord_not_angle_squared(self):
        x = self.fixture
        f = torch.tensor([0, 0])
        S0 = x.ref_state(f)
        init = torch.cat((S0[0], mat_to_quat(S0[1])), -1)
        for theta in (0., math.pi / 2, math.pi - 1e-4, math.pi, math.pi + 1e-4, -math.pi):
            S = list(S0)
            # Right multiplication gives a known relative rotation in each own initial frame.
            delta = so3_exp(torch.tensor([[0., 0., theta]]).expand(2, -1))
            S[1] = S0[1] @ delta
            dr = self.mdp.base_diff(S, f, init, f)[1]
            torch.testing.assert_close(dr.square().sum(-1), torch.full((2,), 4 * math.sin(theta / 2)**2),
                                       atol=2e-6, rtol=1e-5)

    def test_velocities_use_each_current_local_frame(self):
        x = self.fixture
        f = x.f
        S = list(x.transformed(f))
        # Change current tilt as well; this distinguishes current-base and initial/hybrid coordinates.
        S[1] = so3_exp(torch.tensor([[.5, -.4, .7], [-.4, .6, -.8]])) @ S[1]
        local_v = mv3(x.R[f].transpose(-1, -2), x.v[f])
        local_w = mv3(x.R[f].transpose(-1, -2), x.w[f])
        S[2], S[3] = mv3(S[1], local_v), mv3(S[1], local_w)
        dv, dw = self.mdp.base_diff(S, f, x.context(), x.f0)[2:]
        torch.testing.assert_close(dv, torch.zeros(2, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(dw, torch.zeros(2, 3), atol=1e-6, rtol=0)
        self.assertGreater((x.v[f] - S[2]).norm().item(), .1)
        S[2] += mv3(S[1], torch.tensor([[.2, -.3, .1]]).expand(2, -1))
        S[3] += mv3(S[1], torch.tensor([[-.4, .1, .2]]).expand(2, -1))
        _, e = x.reward(S)
        torch.testing.assert_close(e['e_root_linvel'], torch.full((2,), .14), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(e['e_root_angvel'], torch.full((2,), .21), atol=1e-6, rtol=1e-5)

    def test_base_speed_terms_contribute_to_total_reward(self):
        x = self.fixture
        S = list(x.ref_state(x.f))
        S[2], S[3] = S[2] + .2, S[3] - .1
        r, e = self.mdp.reward(S, x.f, x.a, x.a, terms=True, init_root=x.context(), init_f=x.f0)
        c = self.mdp.cfg
        # Reconstruct the complete tracking sum to catch missing base-speed terms or wrong weights.
        n = len(self.mdp.kin.tracked_idx)
        expected = (
            c.w_link_pos * torch.exp(-e['e_body_pos'] * n / c.sigma_link_pos)
            + c.w_link_rotation * torch.exp(-e['e_body_ori'] * n / c.sigma_link_rotation)
            + c.w_link_lin_vel * torch.exp(-e['e_body_linvel'] / (3*c.sigma_link_lin_vel))
            + c.w_link_ang_vel * torch.exp(-e['e_body_angvel'] / (3*c.sigma_link_ang_vel))
            + c.w_base_pos * torch.exp(-e['e_root_pos'] / c.sigma_base_pos)
            + c.w_base_rotation * torch.exp(-e['e_root_ori'] / c.sigma_base_rotation)
            + c.w_base_lin_vel * torch.exp(-e['e_root_linvel'] / c.sigma_base_lin_vel)
            + c.w_base_ang_vel * torch.exp(-e['e_root_angvel'] / c.sigma_base_ang_vel)
            + c.w_dofs_vel * torch.exp(-e['e_joint_vel'] / c.sigma_dofs_vel)
        )
        torch.testing.assert_close(r, expected * CONTROL_DT)
        self.assertAlmostEqual(MAX_STEP_REWARD, 3 * CONTROL_DT)

    def test_local_velocity_gradients_eager_and_compiled(self):
        x = self.fixture
        S = list(x.transformed(x.f))
        S[2] = (S[2] + .2).requires_grad_()
        S[3] = (S[3] - .1).requires_grad_()
        def fn(v, w):
            dv, dw = self.mdp.base_diff((S[0], S[1], v, w, S[4], S[5]), x.f, x.context(), x.f0)[2:]
            return torch.exp(-dv.square().sum(-1)) + torch.exp(-dw.square().sum(-1))
        eager = fn(S[2], S[3])
        compiled = torch.compile(fn, backend='aot_eager', fullgraph=True)(S[2], S[3])
        torch.testing.assert_close(eager, compiled)
        grad = torch.autograd.grad(eager.sum(), (S[2], S[3]))
        for g in grad:
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
