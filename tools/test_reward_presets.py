"""Reference reward weights, sigma conversion, checkpoint round-trip and normalization checks.

Run: python tools/test_reward_presets.py
"""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from blinkmimic.env import TrackingSim
from blinkmimic.mdp import MAX_STEP_REWARD, RewardCfg
from blinkmimic.motion import CONTROL_DT
from blinkmimic.rotations import mat_to_quat, so3_exp
from test_base_reference import inv, mul
from test_body_reference import rotate
import test_body_reference as bodies


class RewardPresetTest(unittest.TestCase):
    def fixture(self, preset):
        x = bodies.BodyReferenceTest()
        x.setUp()
        self.addCleanup(x.doCleanups)
        x.mdp.set_reward_cfg(RewardCfg.from_preset(preset, len(x.kin.tracked_idx)))
        return x

    def test_reference_weights_and_maximum(self):
        c = RewardCfg.from_preset('reference', 14)
        for name, expected in dict(w_root_pos=.5, w_root_ori=.5, w_root_linvel=.2, w_root_angvel=.2,
                                   w_body_pos=.5, w_body_ori=.6, w_body_linvel=.2, w_body_angvel=.2,
                                   w_joint_vel=.1, w_action_rate=-.01, w_action_magnitude=-.01,
                                   w_joint_limit=0.).items():
            self.assertEqual(getattr(c, name), expected)
        self.assertAlmostEqual(c.max_step_reward, 3 * CONTROL_DT)
        self.assertEqual(RewardCfg().max_step_reward, MAX_STEP_REWARD)
        for invalid, n in (('unknown', 14), ('reference', 0)):
            with self.assertRaises(ValueError):
                RewardCfg.from_preset(invalid, n)

    def test_sum_mean_sigma_conversion_for_multiple_link_counts(self):
        torch.manual_seed(13)
        for n in (1, 4, 14):
            c = RewardCfg.from_preset('reference', n)
            d = torch.randn(5, n, 3) * .2
            for std, sigma, aggregation in ((c.body_pos_std, .5, 'sum'), (c.body_ori_std, 1., 'sum'),
                                             (c.body_linvel_std, 1., 'mean'), (c.body_angvel_std, 2., 'mean')):
                ours = torch.exp(-d.square().sum(-1).mean(-1)/std**2)
                ref_error = d.square().sum((1, 2)) if aggregation == 'sum' else d.square().mean((1, 2))
                torch.testing.assert_close(ours, torch.exp(-ref_error/sigma))
        self.assertAlmostEqual(c.root_pos_std**2, .1)
        self.assertAlmostEqual(c.root_ori_std**2, .1)
        self.assertAlmostEqual(c.root_linvel_std**2, 2.)
        self.assertAlmostEqual(c.root_angvel_std**2, 2.)
        self.assertAlmostEqual(c.joint_vel_std**2, 5.)

    def reference_base(self, x, S, init):
        f, m = x.f, x.m
        qr, qm, qr0, qm0 = mat_to_quat(S[1]), mat_to_quat(m.root_R[f]), init[:, 3:], mat_to_quat(m.root_R[f])
        dp = -rotate(inv(qr0), S[0]-init[:, :3])  # init_f == f, so the expert displacement is zero
        err = mul(inv(mul(inv(qr0), qr)), mul(inv(qm0), qm))
        dr = 2 * torch.where(err[:, :1] < 0, -1., 1.) * err[:, 1:]
        dv = rotate(inv(qm), m.root_v[f])-rotate(inv(qr), S[2])
        dw = rotate(inv(qm), m.root_w[f])-rotate(inv(qr), S[3])
        return dp, dr, dv, dw

    def test_full_reward_matches_original_reference_formulas(self):
        x = self.fixture('reference')
        S = list(x.state())
        init = torch.cat((S[0]-.05, mat_to_quat(S[1]@so3_exp(torch.tensor([[.1, 0., 0.]]).expand(2,-1)))), -1)
        a, prev = torch.full((2, 3), .3), torch.full((2, 3), .2)
        actual, terms = x.mdp.reward(S, x.f, a, prev, terms=True, init_root=init, init_f=x.f)
        dp, dr, dv, dw = self.reference_base(x, S, init)
        _, (lp, lr, lv, lw) = x.reference(S)
        expected = (.5*torch.exp(-dp.square().sum(-1)/.1) + .5*torch.exp(-dr.square().sum(-1)/.1)
                    + .2*torch.exp(-dv.square().sum(-1)/2) + .2*torch.exp(-dw.square().sum(-1)/2)
                    + .5*torch.exp(-lp.square().sum((1,2))/.5) + .6*torch.exp(-lr.square().sum((1,2)))
                    + .2*torch.exp(-lv.square().mean((1,2))) + .2*torch.exp(-lw.square().mean((1,2))/2)
                    + .1*torch.exp(-(x.m.qd[x.f]-S[5]).square().sum(-1)/5)
                    - .01*(a-prev).square().sum(-1) - .01*a.square().sum(-1)) * CONTROL_DT
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(terms['action_magnitude'], a.square().sum(-1))
        torch.testing.assert_close(terms['e_joint_vel'], (x.m.qd[x.f]-S[5]).square().sum(-1))

    def test_default_preset_retains_previous_weights_and_penalties(self):
        x = self.fixture('default')
        S = x.state(); a = torch.full((2, 3), .3); prev = a*0
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        r, e = x.mdp.reward(S, x.f, a, prev, terms=True, init_root=init, init_f=x.f)
        c = x.mdp.cfg
        positive = sum(torch.exp(-e[k]/std**2) for k, std in
                       (('e_body_pos',c.body_pos_std),('e_body_ori',c.body_ori_std),
                        ('e_body_linvel',c.body_linvel_std),('e_body_angvel',c.body_angvel_std)))
        positive += .5*sum(torch.exp(-e[k]/std**2) for k, std in
                          (('e_root_pos',c.root_pos_std),('e_root_ori',c.root_ori_std),
                           ('e_root_linvel',c.root_linvel_std),('e_root_angvel',c.root_angvel_std)))
        torch.testing.assert_close(r, (positive-.1*e['action_rate']-2*e['joint_limit'])*CONTROL_DT)

    def test_checkpoint_config_round_trip_and_old_checkpoint_fallback(self):
        c = RewardCfg.from_preset('reference', 14)
        args = json.loads(json.dumps(dict(reward_preset='reference', reward_cfg=asdict(c))))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'ckpt.pt'
            torch.save(dict(args=args), path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(RewardCfg.from_args(loaded['args'], 14), c)
        self.assertEqual(RewardCfg.from_args({}, 14), RewardCfg())
        self.assertEqual(RewardCfg.from_args(dict(reward_preset='reference'), 14), c)
        # Resolved values, rather than a newly selected default, determine evaluation reward.
        args['reward_preset'] = 'default'
        self.assertEqual(RewardCfg.from_args(args, 14), c)

    def test_runtime_normalization_and_sampler_maximum_follow_config(self):
        x = self.fixture('default')
        env = TrackingSim.__new__(TrackingSim)
        env.mdp, env.gamma = x.mdp, .99
        self.assertAlmostEqual(env.G_max, 6*CONTROL_DT/.01)
        x.mdp.set_reward_cfg(RewardCfg.from_preset('reference', 4))
        self.assertAlmostEqual(env.G_max, 3*CONTROL_DT/.01)
        self.assertAlmostEqual(x.mdp.max_step_reward, .06)

    def test_compiled_reference_reward_and_gradients(self):
        x = self.fixture('reference')
        S = list(x.state())
        S[5] = S[5].detach().requires_grad_()
        a = torch.full((2, 3), .3, requires_grad=True)
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        expected = x.mdp.reward(S, x.f, a, a*0, init_root=init, init_f=x.f)
        compiled = torch.compile(x.mdp.reward, backend='aot_eager', fullgraph=True)
        actual = compiled(S, x.f, a, a*0, init_root=init, init_f=x.f)
        torch.testing.assert_close(actual, expected)
        for grad in torch.autograd.grad(actual.sum(), (S[5], a)):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_training_cli_selects_reference_and_disables_extra_contact(self):
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        prefix = source.read_text().split('\nimport torch\n', 1)[0]
        with patch('sys.argv', ['train','--reward_preset','reference','--contact_reward','.25']), redirect_stdout(io.StringIO()):
            ns = {'__file__': str(source)}
            exec(compile(prefix, str(source), 'exec'), ns)
        self.assertEqual(ns['args'].reward_preset, 'reference')
        self.assertEqual(ns['args'].contact_reward, 0.)


if __name__ == '__main__':
    unittest.main()
