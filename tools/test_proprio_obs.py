"""Reference-free history observations: dimensions, sensing, delay, gradients and ONNX export.

Run: python tools/test_proprio_obs.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.mdp import MDP, obs_dim, obs_scale, inv_dim, invariant_features
from blinkmimic.networks import Policy
from blinkmimic.rotations import mv3, so3_exp
from blinkmimic.sensing import SensingCfg, ObsPipeline, ImuCfg, JointVelFilterCfg, obs_noise
import test_body_reference as bodies
import test_random_yaw_eval as evaluation


class Layout:
    def __init__(self, J):
        self.J = J

    def get(self, xi, name):
        if name == 'joint_offset':
            return xi[:, 1:1+self.J]
        if name == 'gyro_bias':
            return torch.zeros(xi.shape[0], 3)
        if name == 'qd_tau_scale':
            return torch.ones(xi.shape[0], self.J)
        raise KeyError(name)


class ProprioObservationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.J, self.B = 29, 4
        J, B = self.J, self.B
        kin = SimpleNamespace(q_lo=-2*torch.ones(J), q_hi=2*torch.ones(J), tracked_idx=[0])
        # No reference rotations at all: observation must be usable without reading expert state.
        self.mdp = MDP(kin, SimpleNamespace(), torch.zeros(J), J+1)
        self.S = (torch.randn(B, 3), so3_exp(torch.randn(B, 3)*.4), torch.randn(B, 3),
                  torch.randn(B, 3), torch.randn(B, J), torch.randn(B, J))
        self.xi, self.a = torch.randn(B, J+1)*.01, torch.randn(B, J)
        self.f = torch.full((B,), 999, dtype=torch.long)

    def pipe(self, cfg):
        robot = SimpleNamespace(num_joints=self.J, joint_names=[f'j{i}' for i in range(self.J)], sensing=cfg)
        return ObsPipeline(robot, self.mdp, Layout(self.J), 'cpu', .005)

    def test_g1_dimensions_noise_scale_and_exact_layout(self):
        self.assertEqual(obs_dim(29), 93)
        obs = self.mdp.obs(self.S, self.a, self.xi, self.f)
        p, R, v, w, q, qd = self.S
        expected = torch.cat((q-self.xi[:, 1:], qd, mv3(R.transpose(-1,-2), w), -R[:, 2], self.a), -1)
        torch.testing.assert_close(obs, expected)
        self.assertEqual(obs.shape, (4, 93))
        self.assertEqual(tuple(obs_scale(29).shape), (93,))
        self.assertEqual(tuple(obs_noise(SensingCfg(), 29).shape), (93,))
        torch.testing.assert_close(obs_scale(29)[-29:], torch.ones(29))
        torch.testing.assert_close(obs_noise(SensingCfg(), 29)[-29:], torch.zeros(29))
        self.assertEqual(invariant_features(self.S).shape[-1], inv_dim(29))  # WM features remain unchanged

    def test_history_and_actions_are_invariant_to_world_yaw_and_reference_frame(self):
        obs = self.mdp.obs(self.S, self.a, self.xi, self.f)
        y = torch.tensor([-2.5, -1.1, .8, 2.1])
        Y = so3_exp(torch.stack((y*0, y*0, y), -1))
        p, R, v, w, q, qd = self.S
        transformed = (mv3(Y, p)+3, Y@R, mv3(Y, v), mv3(Y, w), q, qd)
        other = self.mdp.obs(transformed, self.a, self.xi, self.f+1000)
        torch.testing.assert_close(other, obs, atol=1e-6, rtol=1e-5)
        pi = Policy(93, 17, 29, hist=10, enc_hidden=(8,), dec_hidden=(8,))
        goal = torch.randn(4, 17)
        torch.testing.assert_close(pi(obs[:,None].repeat(1,10,1), goal),
                                   pi(other[:,None].repeat(1,10,1), goal), atol=1e-6, rtol=1e-5)

    def test_raw_sensing_matches_direct_observation_and_pack_shapes(self):
        pipe = self.pipe(SensingCfg())
        st = pipe.init_state(self.S, self.xi, torch.zeros(4,3), torch.zeros(4,3))
        raw = pipe.raw_obs(st, self.S, self.a, self.xi, self.f)
        torch.testing.assert_close(raw, self.mdp.obs(self.S, self.a, self.xi, self.f))
        self.assertEqual(pipe.group.numel(), 93)
        self.assertEqual(pipe.delayed.numel(), 93)
        self.assertEqual(pipe.sizes['prev_raw'], 93)
        packed = pipe.pack(st)
        self.assertEqual(packed.shape[-1], pipe.state_dim)
        for key, val in st.items():
            torch.testing.assert_close(pipe.unpack(packed)[key], val)

    def test_joint_filter_and_delay_keep_previous_actions_exact(self):
        cfg = SensingCfg(delay_hold_prob=1., joint_vel_filter=JointVelFilterCfg(tau={'.*': .05}))
        pipe = self.pipe(cfg)
        st = pipe.init_state(self.S, self.xi, torch.zeros(4,3), torch.zeros(4,3))
        old = pipe.raw_obs(st, self.S, self.a, self.xi, self.f)
        st['prev_raw'], st['lag'] = old, torch.ones(4,4)
        st = pipe.filter_update(st, self.S[5]+1, None, None, self.xi, .02, noise=False, ctrl_rate=True)
        S = list(self.S); S[4] = S[4]+1
        a = self.a+.5
        raw = pipe.raw_obs(st, S, a, self.xi, self.f)
        delayed, next_st = pipe.observe(st, raw, noise=False, delay=True)
        torch.testing.assert_close(delayed[:, :-29], old[:, :-29])
        torch.testing.assert_close(delayed[:, -29:], a)
        self.assertEqual(pipe.pack(next_st).shape[-1], pipe.state_dim)
        noisy, _ = pipe.observe(st, raw, noise=True, delay=False)
        self.assertEqual(noisy.shape, (4,93))
        b = ReplayBuffer(2,4,'cpu',29,30,obs_dim=93,obs_state_dim=pipe.state_dim)
        tr = {k:torch.zeros_like(v[0]) for k,v in b.data.items()}
        tr.update(o=noisy,obs_state=pipe.pack(next_st))
        b.add(tr)
        start, hist = b.starts_at(torch.zeros(4,dtype=torch.long),torch.arange(4),2)
        self.assertEqual(hist['o'].shape,(4,2,93))
        for key,val in next_st.items():
            torch.testing.assert_close(pipe.unpack(start['obs_state'])[key],val)

    def test_imu_sensing_path_matches_proprioception_without_reference(self):
        x = bodies.BodyReferenceTest(); x.setUp(); self.addCleanup(x.doCleanups)
        path = Path(x.tmp.name)/'robot.xml'
        path.write_text(path.read_text().replace('<freejoint/>', '<freejoint/><site name="imu" pos="0 0 .05"/>'))
        cfg = SensingCfg(imu=ImuCfg(site='imu'))
        robot = SimpleNamespace(num_joints=3, joint_names=['a','b','c'], tracked_bodies=['root'], sensing=cfg,
                                mjcf_path=str(path))
        pipe = ObsPipeline(robot, x.mdp, Layout(3), 'cpu', .005)
        S, xi, a = x.state(), torch.zeros(2,4), torch.zeros(2,3)
        gyro = mv3(S[1].transpose(-1,-2), S[3])
        accel = S[1][:,2]*9.81
        st = pipe.init_state(S, xi, gyro, accel)
        # Remove access to motion entirely after state setup.
        x.mdp.motion = None
        torch.testing.assert_close(pipe.raw_obs(st, S, a, xi, x.f), x.mdp.obs(S,a,xi,x.f), atol=1e-6, rtol=1e-5)
        st = pipe.filter_update(st, S[5], gyro+.1, accel, xi, .02, noise=False)
        self.assertEqual(pipe.raw_obs(st,S,a,xi,x.f).shape, (2,15))
        self.assertTrue(torch.isfinite(pipe.pack(st)).all())

    def test_compiled_observation_noise_and_policy_gradients(self):
        w = self.S[3].clone().requires_grad_()
        S = (*self.S[:3], w, *self.S[4:])
        eager = self.mdp.obs(S, self.a, self.xi, self.f)
        compiled = torch.compile(self.mdp.obs, backend='aot_eager', fullgraph=True)(S,self.a,self.xi,self.f)
        torch.testing.assert_close(eager, compiled)
        self.assertEqual(self.mdp.add_noise(compiled).shape, (4,93))
        pi = Policy(93,17,29,hist=2,enc_hidden=(8,),dec_hidden=(8,))
        actions = pi(compiled[:,None].expand(-1,2,-1), torch.randn(4,17))
        grad = torch.autograd.grad(actions.square().sum(), w)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)

    def test_real_onnx_export_metadata_and_reference_inference(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        import export_onnx as exporter
        x = evaluation.RandomYawEvaluationTest(); x.setUp()
        m, K, H = x.source, 2, 2
        pi = Policy(obs_dim(1),m.goal_dim(K),1,hist=H,enc_hidden=(8,),dec_hidden=(8,)).eval()
        robot = SimpleNamespace(num_joints=1, joint_names=['j0'], tracked_bodies=['root'],
                                default_joint_pos_list=lambda:[0.], action_scale_list=lambda:[.5], load_motion=lambda _:None)
        with TemporaryDirectory() as tmp:
            out, ckpt = Path(tmp)/'policy.onnx', Path(tmp)/'ckpt.pt'
            torch.save(dict(pi=pi.state_dict(), args=dict(goal_frames=K,hist=H,enc_hidden='8',dec_hidden='8'),iteration=1),ckpt)
            argv = ['export_onnx.py','--ckpt',str(ckpt),'--motion','dummy.csv','--out',str(out)]
            with patch('sys.argv',argv), patch.object(exporter,'get_robot',return_value=robot), \
                 patch.object(exporter,'Kinematics',return_value=None), patch.object(exporter,'MotionLib',return_value=m), \
                 patch.object(exporter,'pd_gains',return_value=([30.],[1.])):
                exporter.main()
            graph = onnx.load(out)
            onnx.checker.check_model(graph)
            self.assertEqual([d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim], [1,H,9])
            meta = {x.key:x.value for x in graph.metadata_props}
            self.assertEqual(meta['observation_dim'],'9')
            self.assertNotIn('motion_anchor_ori_b',meta['observation_names'])
            model = exporter.PolicyWithMotion(pi,m,K).eval()
            runtime = ReferenceEvaluator(graph)
            obs = torch.randn(1,H,9)
            time = torch.tensor([[5.]])
            with torch.no_grad():
                expected = model(obs,time)
            actual = runtime.run(None,dict(obs_history=obs.numpy(),time_step=time.numpy()))
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b.numpy(),atol=2e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
