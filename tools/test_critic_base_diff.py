"""CPU regression checks for critic base diffs and reset context through policy/value updates.

Run: python tools/test_critic_base_diff.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.mdp import MAX_STEP_REWARD, MDP, flatten_state, unflatten_state
from blinkmimic.networks import Critic, Policy
from blinkmimic.objective import actor_objective
from blinkmimic.rotations import mat_to_quat, mv3, so3_exp
import test_random_yaw_eval as evaluation
from test_relative_reward import RootKinematics


class CriticBaseDiffTest(unittest.TestCase):
    def setUp(self):
        x = evaluation.RandomYawEvaluationTest()
        x.setUp()
        self.motion = x.source
        self.mdp = MDP(RootKinematics(), self.motion, torch.zeros(1), 2)
        self.f0, self.f = torch.tensor([0, 4]), torch.tensor([1, 5])
        yaw = torch.tensor([1.1, -.8])
        self.Y = so3_exp(torch.stack((yaw*0, yaw*0, yaw), -1))
        self.shift = torch.tensor([[3., -2., 0.], [-4., 5., 0.]])
        m = self.motion
        self.init = torch.cat((m.root_p[self.f0] + self.shift, mat_to_quat(self.Y @ m.root_R[self.f0])), -1)
        self.xi = torch.zeros(2, 2)

    def state(self, f):
        m = self.motion
        return (self.init[:, :3] + mv3(self.Y, m.root_p[f] - m.root_p[self.f0]),
                self.Y @ m.root_R[f], mv3(self.Y, m.root_v[f]), mv3(self.Y, m.root_w[f]), m.q[f], m.qd[f])

    def test_exact_layout_and_shared_reward_diffs(self):
        S = list(self.state(self.f))
        S[0] = S[0] + .1
        S[2], S[3] = S[2] + .2, S[3] - .3
        features = self.mdp.critic_features(S, self.f, self.xi, self.init, self.f0)
        offset = 3 * self.mdp.num_j + 3 * len(self.mdp.kin.tracked_idx)
        expected = torch.cat(self.mdp.base_diff(S, self.f, self.init, self.f0), -1)
        torch.testing.assert_close(features[:, offset:offset+12], expected, atol=0, rtol=0)
        self.assertEqual(features.shape, (2, self.mdp.critic_dim))
        self.assertEqual(self.mdp.critic_dim, offset + 12 + self.mdp.xi_dim)
        torch.testing.assert_close(features[:, :2], torch.cat(S[4:], -1))
        _, terms = self.mdp.reward(S, self.f, torch.zeros(2, 1), torch.zeros(2, 1), terms=True,
                                   init_root=self.init, init_f=self.f0)
        for j, key in enumerate(('e_root_pos', 'e_root_ori', 'e_root_linvel', 'e_root_angvel')):
            torch.testing.assert_close(features[:, offset+3*j:offset+3*j+3].square().sum(-1), terms[key])

    def test_no_absolute_yaw_or_position_error_at_ideal_tracking(self):
        actual = self.mdp.critic_features(self.state(self.f), self.f, self.xi, self.init, self.f0)
        m = self.motion
        original_init = torch.cat((m.root_p[self.f0], mat_to_quat(m.root_R[self.f0])), -1)
        original = self.mdp.critic_features(m.ref_state(self.f), self.f, self.xi, original_init, self.f0)
        torch.testing.assert_close(actual, original, atol=2e-6, rtol=1e-5)
        with self.assertRaises(AssertionError):
            self.mdp.critic_features(self.state(self.f), self.f, self.xi, None, None)

    def test_replay_context_survives_reset_and_horizon_expansion(self):
        b = ReplayBuffer(2, 2, 'cpu', 1, self.mdp.xi_dim)
        tr = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
        tr.update(s=flatten_state(self.state(self.f)), f=self.f, init_root=self.init.clone(), init_f=self.f0.clone())
        b.add(tr)
        tr['init_root'].zero_()  # subsequent episode must not overwrite the replay start
        start, _ = b.starts_at(torch.zeros(2, dtype=torch.long), torch.arange(2), 1)
        torch.testing.assert_close(start['init_root'], self.init)
        expand = lambda x: x.unsqueeze(0).expand(3, *x.shape).flatten(0, 1)
        S = tuple(expand(x) for x in unflatten_state(start['s']))
        feat = self.mdp.critic_features(S, expand(start['f']), expand(self.xi),
                                       expand(start['init_root']), expand(start['init_f']))
        one = self.mdp.critic_features(unflatten_state(start['s']), start['f'], self.xi, self.init, self.f0)
        torch.testing.assert_close(feat, expand(one))

    def test_compiled_critic_and_bootstrap_gradients(self):
        S = list(self.state(self.f))
        S[0] = (S[0] + .15).requires_grad_()
        delta = torch.tensor([[.1, -.1, .2], [-.2, .1, -.1]], requires_grad=True)
        S[1] = so3_exp(delta) @ S[1]
        S[2] = (S[2] + .2).requires_grad_()
        S[3] = (S[3] - .1).requires_grad_()
        critic = Critic(self.mdp.critic_dim)
        critic.requires_grad_(False)
        def fn(p, R, v, w):
            return critic(self.mdp.critic_features((p, R, v, w, S[4], S[5]), self.f, self.xi, self.init, self.f0))
        args = tuple(S[:4])
        eager = fn(*args)
        compiled = torch.compile(fn, backend='aot_eager', fullgraph=True)(*args)
        torch.testing.assert_close(eager, compiled)
        for g in torch.autograd.grad(compiled.sum(), (S[0], delta, S[2], S[3])):
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(), 0)

    def test_actual_policy_update_passes_context_to_bootstrap_and_critic_targets(self):
        # Load the actual training functions without executing the train.py CLI/GPU setup.
        source = Path(__file__).parents[1] / 'src/blinkmimic/train.py'
        tree = ast.parse(source.read_text())
        names = {'_critic_input', '_reward_term', '_gae_targets', '_critic_loss', 'policy_update'}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        B, HZ, Hh, K = 2, 2, 2, 2
        pi = Policy(self.mdp.obs_dim, self.motion.goal_dim(K), 1, hist=Hh, enc_hidden=(8,), dec_hidden=(8,))
        critic = Critic(self.mdp.critic_dim + self.motion.goal_dim(K))
        wm = nn.Linear(1, 1)
        s = dict(s=flatten_state(self.state(self.f)), f=self.f, xi=self.xi, a_prev=torch.zeros(B, 1),
                 init_root=self.init, init_f=self.f0)
        h = {k: s[k][:, None].expand(B, Hh, *s[k].shape[1:]) for k in ('s', 'f', 'xi', 'a_prev')}
        args = SimpleNamespace(pi_batch=B, contact_reward=0., gamma=.99, lam=.95, compile_more=True,
                               pi_clip_value=0., no_sync=False, fused_adam=False, critic_epochs=1, critic_minibatches=1)
        ns = dict(torch=torch, nn=nn, mdp=self.mdp, motion=self.motion, pi=pi, critic=critic, wm=wm,
                  args=args, HZ=HZ, Hh=Hh, K=K, GRAPH=False, dev='cpu', MAX_STEP_REWARD=MAX_STEP_REWARD,
                  unflatten_state=unflatten_state, normalize_xi=lambda x: x,
                  sample_valid_starts=lambda n: (s, h), pipe=SimpleNamespace(stateful=False), pi_avg=None,
                  actor_objective=actor_objective, set_requires_grad=lambda model, flag: model.requires_grad_(flag),
                  opt_pi=torch.optim.Adam(pi.parameters(), lr=1e-4), opt_v=torch.optim.Adam(critic.parameters(), lr=1e-4))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
        calls = []
        def critic_input(*values):
            calls.append((values[-2].detach().clone(), values[-1].detach().clone()))
            return ns['_critic_input'](*values)
        def imag_step(S, hist, prev, f, xi, xi_n, g):
            a = pi(hist, g)
            d = torch.cat((a, a*0, a*0), -1)
            S1 = (S[0]+.02*d, S[1], S[2]+.1*d, S[3], S[4]+.01*a, S[5])
            f1 = f+1
            o1 = self.mdp.obs(S1, a, xi, f1)
            return S1, a, torch.cat((hist[:, 1:], o1[:, None]), 1), f1, self.motion.goal(f1, K)
        ns.update(critic_input_c=critic_input, reward_term=ns['_reward_term'], imag_step=imag_step)
        result = ns['policy_update']()
        self.assertTrue(all(torch.isfinite(x).all() for x in result))
        self.assertGreater(result[3].item(), 0)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0][0], self.init)
        torch.testing.assert_close(calls[0][1], self.f0)
        torch.testing.assert_close(calls[1][0], self.init.repeat(HZ+1, 1))
        torch.testing.assert_close(calls[1][1], self.f0.repeat(HZ+1))


if __name__ == '__main__':
    unittest.main()
