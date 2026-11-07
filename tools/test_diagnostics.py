"""CPU checks for reward accounting, replay alignment, model drift and real TensorBoard events.

Run: python tools/test_diagnostics.py
"""

import ast
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.diagnostics import replay_diagnostics
from blinkmimic.mdp import MDP, flatten_state, unflatten_state
from blinkmimic.motion import CONTROL_DT
from blinkmimic.networks import WorldModel
from blinkmimic.rotations import mat_to_quat, mv3, so3_exp
from test_relative_reward import RootKinematics


class ReplayDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        n = 64
        p = torch.zeros(n, 3)
        p[:, 0], p[:, 2] = torch.arange(n) * CONTROL_DT, 1.
        R = torch.eye(3).expand(n, 3, 3).clone()
        v, w = torch.zeros(n, 3), torch.zeros(n, 3)
        v[:, 0] = 1.
        q = torch.zeros(n, 1)
        self.motion = SimpleNamespace(root_p=p, root_R=R, root_v=v, root_w=w, q=q, qd=q,
                                      body_p=p[:, None], body_R=R[:, None], body_v=v[:, None], body_w=w[:, None])
        self.motion.prepare = lambda *a, **kw: None
        self.mdp = MDP(RootKinematics(), self.motion, torch.zeros(1), 1)
        self.mdp.feet = [0]
        self.wm = WorldModel(1, 1, hidden=(8,))
        self.wm.out_norm.mean[0] = CONTROL_DT  # exact constant-speed dynamics in the heading frame
        self.floor = torch.full((14,), 1e-3)
        self.Y = so3_exp(torch.tensor([[0., 0., 1.1], [0., 0., -.7]]))
        self.shift = torch.tensor([[3., -2., 0.], [-5., 4., 0.]])
        self.f0 = torch.tensor([4, 8])
        self.init = torch.cat((p[self.f0] + self.shift, mat_to_quat(self.Y)), -1)
        self.buffer = self.make_buffer()

    def state(self, f, init_f, init):
        m = self.motion
        return (init[:, :3] + mv3(self.Y, m.root_p[f] - m.root_p[init_f]), self.Y,
                mv3(self.Y, m.root_v[f]), m.root_w[f], m.q[f], m.qd[f])

    def make_buffer(self, resets=False):
        b = ReplayBuffer(12, 2, 'cpu', 1, 1)
        f0, init = self.f0.clone(), self.init.clone()
        for t in range(16):  # exercise the circular write head too
            if resets and t == 9:
                f0 = torch.tensor([20, 25])
                init = torch.cat((self.motion.root_p[f0] + 10*self.shift, mat_to_quat(self.Y)), -1)
            f = f0 + (t-9 if resets and t >= 9 else t)
            a, prev = torch.full((2, 1), .3), torch.full((2, 1), .1)
            S1 = self.state(f+1, f0, init)
            r = self.mdp.reward(S1, f+1, a, prev, init_root=init, init_f=f0)
            tr = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
            tr.update(s=flatten_state(self.state(f, f0, init)), s1=flatten_state(S1), a=a, a_prev=prev,
                      f=f, init_root=init, init_f=f0, r=r, ep_step=torch.full((2,), t))
            if resets and t == 8:
                tr['cut'][:] = True
                tr['done'][:] = True
            b.add(tr)
        return b

    def diagnostics(self, buffer=None):
        return replay_diagnostics(buffer or self.buffer, self.wm, self.mdp, lambda x: x, self.floor, 4, 8, batch=32)

    def test_all_weighted_terms_sum_to_reward_including_contact_and_penalties(self):
        f = self.f0+3
        S = self.state(f, self.f0, self.init)
        a = torch.full((2, 1), .3, requires_grad=True)
        for contact in (0., self.mdp.cfg.w_contact):
            r, terms = self.mdp.reward(S, f, a, a*0, terms=True, w_contact=contact,
                                       init_root=self.init, init_f=self.f0)
            self.assertEqual(len([k for k in terms if k.startswith('r_')]), 13)
            self.assertEqual(len([k for k in terms if k.startswith('score_')]), 9)
            torch.testing.assert_close(sum(v for k,v in terms.items() if k.startswith('r_')), r)
            torch.testing.assert_close(torch.autograd.grad(r.sum(), a, retain_graph=True)[0],
                                       torch.autograd.grad(sum(v for k,v in terms.items() if k.startswith('r_')).sum(), a)[0])
            compiled = torch.compile(self.mdp.reward, backend='aot_eager', fullgraph=True)(
                S, f, a, a*0, terms=True, w_contact=contact, init_root=self.init, init_f=self.f0)
            torch.testing.assert_close(compiled, (r, terms))

    def test_exact_model_reward_alignment_reset_frames_and_loss_accounting(self):
        values = self.diagnostics()
        self.assertTrue(all(math.isfinite(v) for v in values.values()))
        self.assertAlmostEqual(values['wm/loss'], 0., places=10)
        self.assertAlmostEqual(values['wm/reward_gap_without_contact'], 0., places=6)
        for source in ('sim', 'model'):
            self.assertAlmostEqual(values[f'score_{source}/base_pos'], 1., places=6)
            self.assertAlmostEqual(values[f'diff_{source}/base_angle_deg'], 0., places=4)
            total = sum(v for k,v in values.items() if k.startswith(f'reward_{source}/') and k.rsplit('/',1)[1] not in ('total', 'tracking'))
            self.assertAlmostEqual(total, values[f'reward_{source}/total'], places=6)
        self.assertGreater(values['reward_model/contact'], 0)
        self.assertEqual(values['reward_sim/contact'], 0)
        self.assertAlmostEqual(values['motion_sim/base_forward_speed'], 1., places=6)

    def test_base_only_model_bias_is_visible_despite_accurate_joints(self):
        self.wm.out_norm.mean[0] = CONTROL_DT-.005
        v = self.diagnostics()
        self.assertGreater(v['wm/loss_contribution/base_pos'], 0)
        self.assertEqual(v['wm/loss_contribution/joint_pos'], 0)
        self.assertEqual(v['wm/loss_contribution/joint_vel'], 0)
        self.assertAlmostEqual(v['wm/rmse_8step/base_pos'], .005*8/math.sqrt(3), places=6)
        self.assertAlmostEqual(v['wm/loss'], sum(x for k,x in v.items() if k.startswith('wm/loss_contribution/')), places=8)
        self.assertLess(v['wm/reward_gap_without_contact'], 0)

    def test_reset_crossing_is_masked_and_rng_and_model_are_unchanged(self):
        b = self.make_buffer(resets=True)
        before_rng = torch.get_rng_state().clone()
        before_wm = {k:v.clone() for k,v in self.wm.state_dict().items()}
        v = self.diagnostics(b)
        torch.testing.assert_close(torch.get_rng_state(), before_rng)
        for k,x in self.wm.state_dict().items():
            torch.testing.assert_close(x, before_wm[k], rtol=0, atol=0)
        self.assertLess(v['wm/valid_fraction'], 1)
        self.assertEqual(v['wm/valid_8step'], 0)
        self.assertTrue(math.isnan(v['wm/rmse_8step/base_pos']))
        self.assertAlmostEqual(v['wm/loss'], 0, places=10)
        self.assertAlmostEqual(v['wm/reward_gap_without_contact'], 0, places=6)

    def test_logged_loss_matches_actual_training_loss_with_masks(self):
        self.wm.out_norm.mean[0] = CONTROL_DT-.005
        self.wm.inc_norm.var[0] = .04
        b = self.make_buffer(resets=True)
        with torch.random.fork_rng():
            w = b.sample_windows(32, 8, extra=('f', 'init_root', 'init_f'))
        S = unflatten_state(w['s'][:, 0])
        preds = []
        for t in range(4):
            S = self.wm.step(S, w['a'][:, t], w['a_prev'][:, t], w['xi'][:, t])
            preds.append(S)
        predicted = tuple(torch.stack([s[i] for s in preds], 1) for i in range(6))
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_wm_err')
        ns = dict(torch=torch, wm=self.wm, ERR_FLOOR=self.floor, unflatten_state=unflatten_state)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        expected, _ = ns['_wm_err'](predicted, w['s1'][:, :4], w['mask'][:, :4])
        self.assertAlmostEqual(self.diagnostics(b)['wm/loss'], float(expected.detach()/4), places=9)

    def test_device_write_head_sampling(self):
        self.buffer.use_device_ptr(device_size=True)
        v = self.diagnostics()
        self.assertAlmostEqual(v['wm/loss'], 0, places=10)
        self.assertAlmostEqual(v['wm/reward_gap_without_contact'], 0, places=6)

    def test_real_event_file_roundtrip_and_resume(self):
        values = self.diagnostics()
        with TemporaryDirectory() as d:
            with SummaryWriter(d) as writer:
                for tag,value in values.items():
                    writer.add_scalar(tag, value, 50)
            with SummaryWriter(d, purge_step=51) as writer:
                writer.add_scalar('wm/loss', .25, 100)
            events = EventAccumulator(d).Reload()
            self.assertEqual([e.step for e in events.Scalars('wm/loss')], [50, 100])
            self.assertEqual(events.Scalars('wm/loss')[-1].step, 100)
            self.assertAlmostEqual(events.Scalars('wm/loss')[-1].value, .25)
            self.assertIn('reward_sim/base_pos', events.Tags()['scalars'])
            self.assertTrue(list(Path(d).glob('events.out.tfevents.*')))


if __name__ == '__main__':
    unittest.main()
