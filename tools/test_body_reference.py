"""Multi-link FK checks against independent quaternion hybrid / rotvec reference equations.

Run: python tools/test_body_reference.py
"""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.rotations import mat_to_quat, mv3, quat_to_mat, so3_exp
from test_base_reference import inv, mul


def rotate(q, v):
    pure = torch.cat((torch.zeros_like(v[..., :1]), v), -1)
    return mul(mul(q, pure), inv(q))[..., 1:]


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    angle = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    zero = angle*0
    return torch.stack(((angle/2).cos(), zero, zero, (angle/2).sin()), -1)


def rotvec(q):
    q = q / q.norm(dim=-1, keepdim=True)
    q = q * torch.where(q[..., :1] < 0, -1., 1.)
    n = q[..., 1:].norm(dim=-1, keepdim=True)
    theta = 2*torch.atan2(n, q[..., :1])
    return torch.where(n < 1e-7, 2*q[..., 1:], theta*q[..., 1:]/n.clamp(min=1e-12))


class BodyReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / 'robot.xml'
        path.write_text('''<mujoco><worldbody><body name="root"><freejoint/>
<geom type="sphere" size=".1"/>
<body name="a" pos=".2 0 .1"><joint name="a" axis="1 0 0" range="-2 2"/>
<geom type="sphere" size=".05"/>
<body name="b" pos="0 .3 .1"><joint name="b" axis="0 1 0" range="-2 2"/>
<geom type="sphere" size=".05"/>
<body name="c" pos=".1 0 .2"><joint name="c" axis="0 0 1" range="-2 2"/>
<geom type="sphere" size=".05"/>
</body></body></body></body></worldbody></mujoco>''')
        robot = SimpleNamespace(mjcf_path=str(path), joint_names=['a', 'b', 'c'], num_joints=3,
                                tracked_bodies=['root', 'a', 'b', 'c'])
        self.kin = Kinematics(robot, 'cpu')
        n = 8
        p = torch.randn(n, 3)
        R = so3_exp(torch.randn(n, 3)*.5)
        v, w = torch.randn(n, 3), torch.randn(n, 3)
        q, qd = torch.randn(n, 3)*.2, torch.randn(n, 3)*.3
        self.m = SimpleNamespace(root_p=p, root_R=R, root_v=v, root_w=w, q=q, qd=qd)
        self.m.ref_state = lambda f: tuple(getattr(self.m, k)[f] for k in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
        self.update_bodies()
        self.mdp = MDP(self.kin, self.m, torch.zeros(3), 4)
        self.f = torch.tensor([2, 6])

    def update_bodies(self):
        m = self.m
        m.body_p, m.body_R, m.body_v, m.body_w = self.kin.forward(m.root_p, m.root_R, m.root_v, m.root_w, m.q, m.qd)

    def state(self):
        m, f = self.m, self.f
        yaw = torch.tensor([1.2, -.9])
        Y = so3_exp(torch.stack((yaw*0, yaw*0, yaw), -1))
        return (m.root_p[f]+torch.tensor([2., -3., .2]),
                Y @ m.root_R[f] @ so3_exp(torch.tensor([[.3, -.2, .1], [-.2, .4, -.1]])),
                mv3(Y, m.root_v[f])+.1, mv3(Y, m.root_w[f])-.1, m.q[f]+.15, m.qd[f]-.1)

    def reference(self, S):
        f, m = self.f, self.m
        bp, bR, bv, bw = self.kin.forward(*S)
        qm, qr = mat_to_quat(m.root_R[f]), mat_to_quat(S[1])
        qh = mul(yaw_quat(qr), mul(inv(yaw_quat(qm)), qm))
        iqm, iqh = inv(qm)[:, None], inv(qh)[:, None]
        dp = rotate(iqm, m.body_p[f]-m.root_p[f, None])-rotate(iqh, bp-S[0][:, None])
        local_ref = mul(iqm, mat_to_quat(m.body_R[f]))
        local_robot = mul(iqh, mat_to_quat(bR))
        dr = rotvec(mul(inv(local_robot), local_ref))
        dv = rotate(iqm, m.body_v[f])-rotate(iqh, bv)
        dw = rotate(iqm, m.body_w[f])-rotate(iqh, bw)
        return qh, (dp, dr, dv, dw)

    def test_hybrid_and_all_diffs_match_quaternion_reference(self):
        S = self.state()
        qh, expected = self.reference(S)
        H = self.mdp.hybrid_root(S[1], self.f)
        torch.testing.assert_close(H, quat_to_mat(qh), atol=1e-6, rtol=1e-5)
        for actual, ref in zip(self.mdp.body_diff(S, self.f), expected):
            self.assertEqual(actual.shape, (2, 4, 3))
            torch.testing.assert_close(actual, ref, atol=2e-6, rtol=1e-5)

    def test_independent_translation_and_robot_yaw_do_not_create_body_error(self):
        m, f = self.m, self.f
        y = torch.tensor([1.5, -1.4])
        Y = so3_exp(torch.stack((y*0, y*0, y), -1))
        S = (m.root_p[f]+torch.tensor([3., -4., .7]), Y@m.root_R[f],
             mv3(Y, m.root_v[f]), mv3(Y, m.root_w[f]), m.q[f], m.qd[f])
        for diff in self.mdp.body_diff(S, f):
            torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-6, rtol=0)
        # No root height term remains hidden inside body position errors.
        self.assertTrue((S[0][:, 2]-m.root_p[f, 2] > .6).all())

    def test_independent_expert_and_robot_world_headings_preserve_body_diffs(self):
        S = self.state()
        expected = self.mdp.body_diff(S, self.f)
        Z = so3_exp(torch.tensor([0., 0., -1.8]))
        Y = so3_exp(torch.tensor([0., 0., 2.1]))
        m = self.m
        m.root_p = mv3(Z, m.root_p) + torch.tensor([-2., 4., 1.])
        m.root_R = Z @ m.root_R
        m.root_v, m.root_w = mv3(Z, m.root_v), mv3(Z, m.root_w)
        self.update_bodies()
        transformed = (mv3(Y, S[0])+torch.tensor([5., -3., -.5]), Y@S[1],
                       mv3(Y, S[2]), mv3(Y, S[3]), S[4], S[5])
        for actual, previous in zip(self.mdp.body_diff(transformed, self.f), expected):
            torch.testing.assert_close(actual, previous, atol=2e-6, rtol=2e-5)

    def test_tilt_error_is_preserved_in_hybrid_velocities(self):
        self.m.root_R[:] = torch.eye(3)
        self.m.root_v[:] = torch.tensor([1., 0., 0.])
        self.m.root_w[:] = torch.tensor([0., 1., 0.])
        self.update_bodies()
        f, m = self.f, self.m
        R = so3_exp(torch.tensor([[0., math.pi/6, 0.]]).expand(2, -1))
        S = (m.root_p[f], R, mv3(R, m.root_v[f]), mv3(R, m.root_w[f]), m.q[f], m.qd[f])
        diff = self.mdp.body_diff(S, f)
        self.assertGreater(diff[2][:, 0].square().sum().item(), .5)
        torch.testing.assert_close(diff[2][:, 0], m.root_v[f]-S[2], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(mv3(R.transpose(-1,-2), S[2]), m.root_v[f])

    def test_rotvec_angle_zero_ninety_and_near_pi(self):
        m = self.m
        m.root_R[:] = torch.eye(3)
        m.q.zero_()
        self.update_bodies()
        f = self.f
        for angle in (0., math.pi/2, math.pi-1e-4, math.pi, math.pi+1e-4):
            q = m.q[f].clone(); q[:, 0] = angle
            q.requires_grad_()
            S = (m.root_p[f], m.root_R[f], m.root_v[f], m.root_w[f], q, m.qd[f])
            dr = self.mdp.body_diff(S, f)[1]
            theta = min(angle, 2*math.pi-angle)
            torch.testing.assert_close(dr[:, 0], torch.zeros(2, 3), atol=1e-6, rtol=0)
            torch.testing.assert_close(dr[:, 1:].norm(dim=-1), torch.full((2, 3), theta), atol=2e-5, rtol=1e-5)
            self.assertTrue(torch.isfinite(dr).all())
            grad = torch.autograd.grad(dr.square().sum(), q)[0]
            self.assertTrue(torch.isfinite(grad).all())
        # Three moving links plus the root: choose theta so summed link angle^2 equals sigma.
        theta = math.sqrt(self.mdp.cfg.sigma_link_rotation / 3)
        q = m.q[f].clone(); q[:, 0] = theta
        S = (m.root_p[f], m.root_R[f], m.root_v[f], m.root_w[f], q, m.qd[f])
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        _, terms = self.mdp.reward(S, f, torch.zeros(2, 3), torch.zeros(2, 3), terms=True, init_root=init, init_f=f)
        reward_rot = torch.exp(-4*terms['e_body_ori']/self.mdp.cfg.sigma_link_rotation)
        torch.testing.assert_close(reward_rot, torch.full((2,), math.exp(-1)), atol=1e-6, rtol=1e-5)

    def test_reward_aggregation_sigma_and_critic_share_every_diff(self):
        S, f = self.state(), self.f
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        _, terms = self.mdp.reward(S, f, torch.zeros(2, 3), torch.zeros(2, 3), terms=True, init_root=init, init_f=f)
        diffs = self.mdp.body_diff(S, f)
        for diff, key in zip(diffs, ('e_body_pos', 'e_body_ori', 'e_body_linvel', 'e_body_angvel')):
            torch.testing.assert_close(terms[key], diff.square().sum(-1).mean(-1))
        feat = self.mdp.critic_features(S, f, torch.zeros(2, 4), init, f)
        body = torch.cat([d.flatten(-2) for d in diffs], -1)
        torch.testing.assert_close(feat[:, 9:9+48], body, atol=0, rtol=0)
        # Diagnostic mean errors stay comparable when links are duplicated. The configured
        # reference position/rotation reward intentionally sums links; velocity reward averages them.
        expected = {k: terms[k] for k in ('e_body_pos', 'e_body_ori', 'e_body_linvel', 'e_body_angvel')}
        self.kin.tracked_idx *= 2
        self.update_bodies()
        doubled = MDP(self.kin, self.m, torch.zeros(3), 4)
        _, actual = doubled.reward(S, f, torch.zeros(2, 3), torch.zeros(2, 3), terms=True, init_root=init, init_f=f)
        for k in expected:
            torch.testing.assert_close(actual[k], expected[k])

    def test_vertical_heading_is_a_valid_rotation_with_finite_gradient(self):
        R = torch.tensor([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]]).expand(2, -1, -1).clone().requires_grad_()
        hybrid = self.mdp.hybrid_root(R, self.f)
        torch.testing.assert_close(hybrid.transpose(-1,-2)@hybrid, torch.eye(3).expand(2,-1,-1), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(torch.linalg.det(hybrid), torch.ones(2), atol=1e-6, rtol=1e-5)
        hybrid.sum().backward()
        self.assertTrue(torch.isfinite(R.grad).all())

    def test_compiled_diffs_and_gradients(self):
        S = list(self.state())
        for i in (0, 1, 2, 3, 4, 5):
            S[i] = S[i].detach().requires_grad_()
        eager = self.mdp.body_diff(S, self.f)
        compiled = torch.compile(self.mdp.body_diff, backend='aot_eager', fullgraph=True)(S, self.f)
        for a, b in zip(eager, compiled):
            torch.testing.assert_close(a, b)
        import blinkmimic.rotations as rotations
        old = rotations.ELEMENTWISE
        rotations.ELEMENTWISE = True
        try:
            for a, b in zip(eager, self.mdp.body_diff(S, self.f)):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
        finally:
            rotations.ELEMENTWISE = old
        loss = sum(x.square().mean() for x in compiled)
        for i, g in enumerate(torch.autograd.grad(loss, S)):
            self.assertTrue(torch.isfinite(g).all())
            if i == 0:  # translating the entire robot cancels after subtracting its own root
                torch.testing.assert_close(g, torch.zeros_like(g), atol=1e-6, rtol=0)
            else:
                self.assertGreater(g.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
