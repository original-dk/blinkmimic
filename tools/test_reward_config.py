"""Checks for the single reward file, direct reference sigma formulas and gradients.

Run: python tools/test_reward_config.py
"""

from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import CONTROL_DT
from blinkmimic.reward_cfg import REWARD_CFG, RewardCfg
from blinkmimic.rotations import mat_to_quat, so3_exp
from test_base_reference import inv, mul
from test_body_reference import rotate
import test_body_reference as bodies


class RewardConfigTest(unittest.TestCase):
    def fixture(self):
        x = bodies.BodyReferenceTest()
        x.setUp()
        self.addCleanup(x.doCleanups)
        return x

    def test_active_values_match_preceding_repository_configuration(self):
        c = REWARD_CFG
        for name, expected in dict(sigma_base_pos=.3**2, sigma_base_rotation=.4**2,
                                   sigma_base_lin_vel=1., sigma_base_ang_vel=3.14**2,
                                   sigma_link_pos=14*.3**2, sigma_link_rotation=14*.4**2,
                                   sigma_link_lin_vel=1/3, sigma_link_ang_vel=3.14**2/3,
                                   w_base_pos=.5, w_base_rotation=.5, w_base_lin_vel=.5, w_base_ang_vel=.5,
                                   w_link_pos=1., w_link_rotation=1., w_link_lin_vel=1., w_link_ang_vel=1.,
                                   w_dofs_vel=0., w_action_rate=-.1, w_action_magnitude=0.,
                                   w_joint_limit=-2., w_contact=.25).items():
            self.assertEqual(getattr(c, name), expected)
        self.assertAlmostEqual(MAX_STEP_REWARD, 6*CONTROL_DT)

    def test_every_mdp_uses_the_same_configuration(self):
        x = self.fixture()
        other = MDP(x.kin, x.m, torch.zeros(3), 4)
        self.assertIs(x.mdp.cfg, REWARD_CFG)
        self.assertIs(other.cfg, REWARD_CFG)

    def reference_base(self, x, S, init):
        qm, qr = mat_to_quat(x.m.root_R[x.f]), mat_to_quat(S[1])
        qr0 = init[:, 3:]
        dp = -rotate(inv(qr0), S[0]-init[:, :3])  # init_f == f
        err = mul(inv(mul(inv(qr0), qr)), mul(inv(qm), qm))
        dr = 2 * torch.where(err[:, :1] < 0, -1., 1.) * err[:, 1:]
        dv = rotate(inv(qm), x.m.root_v[x.f])-rotate(inv(qr), S[2])
        dw = rotate(inv(qm), x.m.root_w[x.f])-rotate(inv(qr), S[3])
        return dp, dr, dv, dw

    def test_complete_reward_matches_independent_reference_equations(self):
        x = self.fixture()
        S = list(x.state())
        init = torch.cat((S[0]-.05, mat_to_quat(S[1]@so3_exp(torch.tensor([[.1, 0., 0.]]).expand(2,-1)))), -1)
        a, prev = torch.full((2, 3), .3), torch.full((2, 3), .2)
        actual, terms = x.mdp.reward(S, x.f, a, prev, terms=True, init_root=init, init_f=x.f)
        dp, dr, dv, dw = self.reference_base(x, S, init)
        _, (lp, lr, lv, lw) = x.reference(S)
        c = REWARD_CFG
        expected = (c.w_base_pos*torch.exp(-dp.square().sum(-1)/c.sigma_base_pos)
                    + c.w_base_rotation*torch.exp(-dr.square().sum(-1)/c.sigma_base_rotation)
                    + c.w_base_lin_vel*torch.exp(-dv.square().sum(-1)/c.sigma_base_lin_vel)
                    + c.w_base_ang_vel*torch.exp(-dw.square().sum(-1)/c.sigma_base_ang_vel)
                    + c.w_link_pos*torch.exp(-lp.square().sum((1,2))/c.sigma_link_pos)
                    + c.w_link_rotation*torch.exp(-lr.square().sum((1,2))/c.sigma_link_rotation)
                    + c.w_link_lin_vel*torch.exp(-lv.square().mean((1,2))/c.sigma_link_lin_vel)
                    + c.w_link_ang_vel*torch.exp(-lw.square().mean((1,2))/c.sigma_link_ang_vel)
                    + c.w_dofs_vel*torch.exp(-(x.m.qd[x.f]-S[5]).square().sum(-1)/c.sigma_dofs_vel)
                    + c.w_action_rate*(a-prev).square().sum(-1) + c.w_action_magnitude*a.square().sum(-1)
                    + c.w_joint_limit*terms['joint_limit']) * CONTROL_DT
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(terms['action_magnitude'], a.square().sum(-1))
        torch.testing.assert_close(terms['e_joint_vel'], (x.m.qd[x.f]-S[5]).square().sum(-1))

    def test_compiled_reward_and_action_gradients(self):
        x = self.fixture()
        S = list(x.state())
        S[5] = S[5].detach().requires_grad_()
        a = torch.full((2, 3), .3, requires_grad=True)
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        expected = x.mdp.reward(S, x.f, a, a*0, init_root=init, init_f=x.f)
        actual = torch.compile(x.mdp.reward, backend='aot_eager', fullgraph=True)(S, x.f, a, a*0, init_root=init, init_f=x.f)
        torch.testing.assert_close(actual, expected)
        for grad in torch.autograd.grad(actual.sum(), (S[5], a)):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_training_has_no_reward_selector_or_contact_override(self):
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        prefix = source.read_text().split('\nimport torch\n', 1)[0]
        for flag, value in (('--reward_preset', 'reference'), ('--contact_reward', '.25')):
            with patch('sys.argv', ['train', flag, value]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ended:
                exec(compile(prefix, str(source), 'exec'), {'__file__': str(source)})
            self.assertEqual(ended.exception.code, 2)

    def test_commented_reference_remains_available(self):
        source = Path(__file__).parents[1]/'src/blinkmimic/reward_cfg.py'
        self.assertIn('# REWARD_CFG = RewardCfg()', source.read_text())
        reference = RewardCfg()
        self.assertEqual(reference.sigma_base_pos, .1)
        self.assertEqual(reference.w_action_rate, -.01)
        self.assertEqual(reference.w_dofs_vel, .1)


if __name__ == '__main__':
    unittest.main()
