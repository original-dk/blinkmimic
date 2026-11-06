"""CPU checks for randomized evaluation references, observations and recorded ghost poses.

Run: python tools/test_random_yaw_eval.py
"""

import math
import unittest

import torch

from blinkmimic.evaluateran import reset_random_yaw, sample_starts
from blinkmimic.env import sample_reset_xy
from blinkmimic.mdp import MDP, RewardCfg
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import mat_to_quat, mv3, so3_exp
import test_relative_reward as relative
from test_relative_reward import RootKinematics


class RandomYawEvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        m = MotionLib.__new__(MotionLib)
        m.device, m.names, m.num_frames, m.num_bins = 'cpu', ['a', 'b'], 8, 2
        m.clip_start, m.clip_end, m.clip_len = torch.tensor([0, 4]), torch.tensor([3, 7]), torch.tensor([4, 4])
        m.clip_id = torch.arange(2).repeat_interleave(4)
        m.bin_start, m.bin_clip, m.frame_bin = m.clip_start.clone(), torch.arange(2), m.clip_id.clone()
        m.root_p = torch.randn(8, 3)
        m.root_R = so3_exp(torch.randn(8, 3) * .3)
        m.root_v, m.root_w = torch.randn(8, 3), torch.randn(8, 3)
        m.q, m.qd = torch.zeros(8, 1), torch.randn(8, 1)
        m.body_p, m.body_R = m.root_p[:, None], m.root_R[:, None]
        m.body_v, m.body_w = m.root_v[:, None], m.root_w[:, None]
        m.root_v_local = mv3(m.root_R.transpose(-1, -2), m.root_v)
        m.root_w_local = mv3(m.root_R.transpose(-1, -2), m.root_w)
        m.root_gz = m.root_R[:, 2, 2]
        self.source = m

    def test_sampling_seed_bounds_and_all_valid_frames(self):
        before = torch.get_rng_state().clone()
        a = sample_starts(self.source, 8192, math.pi / 2, 7)
        b = sample_starts(self.source, 8192, math.pi / 2, 7)
        torch.testing.assert_close(torch.get_rng_state(), before)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
        clips, frames, yaw = a
        self.assertEqual(set(clips.tolist()), {0, 1})
        self.assertEqual(set(frames.tolist()), {0, 1, 2, 4, 5, 6})
        self.assertTrue((frames >= self.source.clip_start[clips]).all())
        self.assertTrue((frames < self.source.clip_end[clips]).all())
        self.assertTrue((yaw.abs() <= math.pi / 2).all())
        self.assertFalse(torch.equal(frames, sample_starts(self.source, 8192, math.pi / 2, 8)[1]))

    def test_short_clips_and_disabled_yaw(self):
        self.source.clip_len = torch.tensor([1, 4])
        clips, frames, yaw = sample_starts(self.source, 100, 0, 0)
        self.assertTrue((clips == 1).all())
        torch.testing.assert_close(yaw, torch.zeros(100))
        self.source.clip_len.fill_(1)
        with self.assertRaises(ValueError):
            sample_starts(self.source, 1, 1, 0)
        with self.assertRaises(ValueError):
            sample_starts(self.source, 0, 1, 0)

    def make_env(self):
        fixture = relative.RelativeRewardTest()
        fixture.setUp()
        e = fixture.fake_env()
        e.motion = self.source
        e.mdp = MDP(RootKinematics(), self.source, torch.zeros(1), 1)
        return e

    def test_robot_only_yaw_reset_preserves_original_commands_and_reference(self):
        e = self.make_env()
        frames, yaw = torch.tensor([2, 5]), torch.tensor([1.2, -1.4])
        before = {k: v.clone() for k, v in vars(self.source).items() if isinstance(v, torch.Tensor)}
        commands = self.source.goal(frames, 5).clone()
        reset_random_yaw(e, frames, yaw)
        self.assertIs(e.motion, self.source)
        self.assertIs(e.mdp.motion, self.source)
        torch.testing.assert_close(self.source.goal(frames, 5), commands, atol=0, rtol=0)
        for k, v in before.items():
            torch.testing.assert_close(getattr(self.source, k), v, atol=0, rtol=0)
        Y = so3_exp(torch.stack((yaw*0, yaw*0, yaw), -1))
        pose = e.robot.write_root_link_pose_to_sim.call_args.args[0]
        vel = e.robot.write_root_link_velocity_to_sim.call_args.args[0]
        torch.testing.assert_close(pose[:, :3], self.source.root_p[frames])
        torch.testing.assert_close(pose[:, 3:], mat_to_quat(Y @ self.source.root_R[frames]))
        torch.testing.assert_close(vel[:, :3], mv3(Y, self.source.root_v[frames]))
        torch.testing.assert_close(vel[:, 3:], mv3(Y, self.source.root_w[frames]))
        torch.testing.assert_close(e.init_root, pose)
        torch.testing.assert_close(e.init_f, frames)
        S = (pose[:, :3], Y @ self.source.root_R[frames], vel[:, :3], vel[:, 3:],
             self.source.q[frames], self.source.qd[frames])
        _, terms = e.mdp.reward(S, frames, torch.zeros(2, 1), torch.zeros(2, 1), terms=True,
                                init_root=e.init_root, init_f=e.init_f)
        for k in ('e_root_pos', 'e_root_ori', 'e_body_linvel', 'e_body_angvel'):
            torch.testing.assert_close(terms[k], torch.zeros(2), atol=3e-6, rtol=0)
        # Observation keeps the original reference too, matching the training path.
        obs = e.mdp.obs(S, torch.zeros(2, 1), torch.zeros(2, 1), frames)
        base_obs = e.mdp.obs(self.source.ref_state(frames), torch.zeros(2, 1), torch.zeros(2, 1), frames)
        self.assertGreater((obs - base_obs).abs().max().item(), .1)

    def test_translation_and_yaw_reset_keep_commands_and_relative_position_reward(self):
        e = self.make_env()
        frames, yaw = torch.tensor([2, 5]), torch.tensor([1.2, -1.4])
        offset = torch.tensor([[3., -2., 0.], [-4., 5., 0.]])
        command = self.source.goal(frames, 5).clone()
        reset_random_yaw(e, frames, yaw, offset)
        torch.testing.assert_close(e.init_root[:, :3], self.source.root_p[frames] + offset)
        torch.testing.assert_close(self.source.goal(frames, 5), command, atol=0, rtol=0)
        Y = so3_exp(torch.stack((yaw*0, yaw*0, yaw), -1))
        f = frames + 1
        p = e.init_root[:, :3] + mv3(Y, self.source.root_p[f] - self.source.root_p[frames])
        S = (p, Y @ self.source.root_R[f], mv3(Y, self.source.root_v[f]), mv3(Y, self.source.root_w[f]),
             self.source.q[f], self.source.qd[f])
        a = torch.zeros(2, 1)
        _, terms = e.mdp.reward(S, f, a, a, terms=True, init_root=e.init_root, init_f=frames)
        torch.testing.assert_close(terms['e_root_pos'], torch.zeros(2), atol=1e-10, rtol=0)

    def test_xy_sampling_bounds_seed_height_and_compile(self):
        p = torch.randn(1024, 3)
        rng = torch.Generator().manual_seed(123)
        offset = sample_reset_xy(p, (-3., 4.), rng)
        again = sample_reset_xy(p, (-3., 4.), torch.Generator().manual_seed(123))
        torch.testing.assert_close(offset, again, atol=0, rtol=0)
        self.assertTrue((offset[:, :2] >= -3).all() and (offset[:, :2] <= 4).all())
        self.assertLess(offset[:, :2].min().item(), -2.9)
        self.assertGreater(offset[:, :2].max().item(), 3.9)
        torch.testing.assert_close(offset[:, 2], torch.zeros(1024), atol=0, rtol=0)
        torch.testing.assert_close(sample_reset_xy(p, (0., 0.)), torch.zeros_like(p), atol=0, rtol=0)
        compiled = torch.compile(sample_reset_xy, backend='aot_eager', fullgraph=True)
        torch.testing.assert_close(compiled(p, (.2, .2))[:, :2], torch.full((1024, 2), .2))

    def test_indexed_and_masked_training_translation_preserve_other_worlds(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from blinkmimic.dr import DomainRandCfg
        a, b = self.make_env(), self.make_env()
        for e in (a, b):
            e.reset_noise = True
            e.spec.dr = DomainRandCfg(reset_xy=(.2, .2), reset_pos=(0.,)*3, reset_rot=(0.,)*3,
                                     reset_yaw=0., reset_lin_vel=(0.,)*3, reset_ang_vel=(0.,)*3,
                                     reset_joint_pos=0.)
        frames = torch.tensor([2, 5])
        a.reset(torch.arange(2), frames)
        with patch('blinkmimic.env.wp.ScopedDevice', return_value=nullcontext()), \
             patch('blinkmimic.env.mjwarp.reset_data'):
            b.reset_masked(torch.ones(2, dtype=torch.bool), frames)
        expected = self.source.root_p[frames] + torch.tensor([.2, .2, 0.])
        torch.testing.assert_close(a.init_root[:, :3], expected)
        torch.testing.assert_close(a.init_root, b.init_root)
        before = b.init_root.clone()
        a.reset(torch.tensor([0]), torch.tensor([1]))
        with patch('blinkmimic.env.wp.ScopedDevice', return_value=nullcontext()), \
             patch('blinkmimic.env.mjwarp.reset_data'):
            b.reset_masked(torch.tensor([True, False]), torch.tensor([1, 4]))
        torch.testing.assert_close(a.init_root, b.init_root)
        torch.testing.assert_close(b.init_root[1], before[1], atol=0, rtol=0)

    def test_zero_yaw_and_remaining_return_normalization(self):
        e = self.make_env()
        frames = torch.tensor([2, 6])  # penultimate frames: exactly one transition remains
        reset_random_yaw(e, frames, torch.zeros(2))
        pose = e.robot.write_root_link_pose_to_sim.call_args.args[0]
        torch.testing.assert_close(pose[:, 3:], mat_to_quat(self.source.root_R[frames]))
        remaining = self.source.clip_end[self.source.clip_id[frames]] - frames
        torch.testing.assert_close(remaining, torch.ones(2, dtype=torch.long))

    def test_evaluator_loop_records_original_commands_and_remaining_returns(self):
        self.check_evaluator_loop('default')

    def test_evaluator_loads_reference_preset_and_normalizes_by_its_maximum(self):
        self.check_evaluator_loop('reference')

    def check_evaluator_loop(self, preset):
        from contextlib import ExitStack, redirect_stdout
        import io
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        from blinkmimic.evaluateran import main
        from blinkmimic.mdp import MAX_STEP_REWARD

        source = self.source
        instances, commands = [], []

        class Env:
            def __init__(self, cfg, n, robot, motion, mdp, dev, **kwargs):
                instances.append(self)
                self.N, self.motion, self.mdp = n, motion, mdp
                self.done_eval = torch.zeros(n, dtype=torch.bool)
                self.failed_eval = torch.zeros(n, dtype=torch.bool)
                self.init_root = torch.zeros(n, 7)
                self.scene = SimpleNamespace(env_origins=torch.zeros(n, 3))
                self.robot = Mock()
                self.robot.write_root_link_pose_to_sim.side_effect = lambda pose, **kw: setattr(self, 'pose', pose)
                self.robot.write_root_link_velocity_to_sim.side_effect = lambda vel, **kw: setattr(self, 'vel', vel)

            def reset(self, ids, frames):
                self.f = frames.clone()
                self.init_f = frames.clone()

            def state(self):
                return (self.pose[:, :3], so3_exp(torch.zeros(self.N, 3)), self.vel[:, :3], self.vel[:, 3:],
                        self.motion.q[self.f], self.motion.qd[self.f])

            def observe(self):
                return torch.zeros(self.N, self.mdp.obs_dim)

            def step(self, a):
                self.f = self.f + 1
                self.done_eval |= self.f >= self.motion.clip_end[self.motion.clip_id[self.init_f]]
                return dict(r=torch.full((self.N,), self.mdp.max_step_reward),
                            terms=dict(e_root_pos=torch.full((self.N,), 0.04)))

        class Actor:
            def __init__(self, *args, **kwargs):
                pass

            def to(self, dev):
                return self

            def load_state_dict(self, state):
                pass

            def eval(self):
                pass

            def __call__(self, hist, goal):
                f = instances[-1].f
                torch.testing.assert_close(goal, source.goal(f, 2), atol=0, rtol=0)
                commands.append(goal.clone())
                return torch.zeros(len(f), 1)

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            record = str(Path(tmp) / 'traj.npz')
            robot = SimpleNamespace(num_joints=1, sensing=None, default_joint_pos_list=lambda: [0.])
            stack.enter_context(patch('sys.argv', ['evaluateran', '--run_dir', tmp, '--ckpt', 'dummy.pt',
                                                  '--record', record, '--episodes', '12', '--seed', '4']))
            stack.enter_context(patch('blinkmimic.robot.get_robot', return_value=robot))
            stack.enter_context(patch('blinkmimic.kinematics.Kinematics', return_value=RootKinematics()))
            stack.enter_context(patch('blinkmimic.motion.MotionLib.from_dir', return_value=source))
            robot.motion_dir = ''
            from blinkmimic.sensing import SensingCfg
            robot.sensing = SensingCfg()
            stack.enter_context(patch('blinkmimic.dr.XiLayout', return_value=SimpleNamespace(dim=1)))
            stack.enter_context(patch('blinkmimic.env.make_sim', return_value=None))
            stack.enter_context(patch('blinkmimic.env.TrackingSim', Env))
            stack.enter_context(patch('blinkmimic.networks.Policy', Actor))
            stack.enter_context(patch('blinkmimic.networks.policy_kwargs', return_value={}))
            stack.enter_context(patch('torch.load', return_value=dict(args=dict(goal_frames=2, hist=1, reward_preset=preset), pi={})))
            # Run the real CLI/evaluation/recording logic with CPU simulation and policy doubles.
            for name in ('tensor', 'zeros', 'ones', 'arange'):
                original = getattr(torch, name)
                def cpu_factory(*args, _fn=original, **kwargs):
                    if kwargs.get('device') == 'cuda:0':
                        kwargs['device'] = 'cpu'
                    return _fn(*args, **kwargs)
                stack.enter_context(patch.object(torch, name, cpu_factory))
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as ended:
                main()
            self.assertEqual(ended.exception.code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(instances[-1].mdp.cfg, RewardCfg.from_preset(preset, 1))
            self.assertAlmostEqual(result['norm_return'], 1, places=6)
            self.assertAlmostEqual(result['norm_return_framew'], 1, places=6)
            self.assertEqual(sum(result['per_clip_count'].values()), 12)
            self.assertAlmostEqual(result['root_pos_rmse_m'], .2, places=6)
            self.assertAlmostEqual(result['root_pos_max_m'], .2, places=6)
            np.testing.assert_allclose(result['per_rollout_root_pos_final_m'], .2, atol=1e-6)
            self.assertTrue(commands)
            d = dict(np.load(record))
            np.testing.assert_array_equal(d['ref_root_p'], source.root_p.numpy())
            np.testing.assert_array_equal(d['f'][0], d['start_frame'])
            np.testing.assert_array_equal(source.clip_id[d['start_frame']].numpy(), d['rollout_clip'])
            self.assertEqual(d['root_p'].shape[1], 12)
            self.assertTrue(((d['pos_offset'][:, :2] >= 0) & (d['pos_offset'][:, :2] <= .1)).all())
            np.testing.assert_array_equal(d['pos_offset'][:, 2], 0)
            np.testing.assert_allclose(d['init_root'][:, :3],
                                       source.root_p[d['start_frame']].numpy() + d['pos_offset'])
            np.testing.assert_allclose(d['root_p'][0], d['init_root'][:, :3])

    def test_long_trajectory_ghost_distance_equals_reward_base_error(self):
        import importlib.util
        from pathlib import Path
        import numpy as np
        spec = importlib.util.spec_from_file_location('render_ran', Path(__file__).with_name('render_ran.py'))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        # Arbitrary nonzero reset frame, tilted initial base and a curved 20 m trajectory.
        m = self.source
        t = torch.linspace(0, 1, 1001)
        m.root_p = torch.stack((20*t, 4*torch.sin(3*t), 1+.1*t), -1)
        m.root_R = so3_exp(torch.stack((.2+.1*t, -.3+.05*t, 2*t), -1))
        m.root_v = torch.stack((torch.full_like(t, 20.), 12*torch.cos(3*t), torch.full_like(t, .1)), -1)
        m.root_w = torch.zeros_like(m.root_v)
        m.q, m.qd = torch.zeros(len(t), 1), torch.zeros(len(t), 1)
        m.body_p, m.body_R = m.root_p[:, None], m.root_R[:, None]
        m.body_v, m.body_w = m.root_v[:, None], m.root_w[:, None]
        f0 = 137
        f = torch.arange(f0, len(t))
        mdp = MDP(RootKinematics(), m, torch.zeros(1), 1)
        for yaw in (-1.57, -.73, 0., .82, 1.57):
            Y = so3_exp(torch.tensor([0., 0., yaw]))
            p0 = m.root_p[f0] + torch.tensor([3., -2., 0.])
            ideal_p = p0 + mv3(Y, m.root_p[f] - m.root_p[f0])
            R = Y @ m.root_R[f]
            init = torch.cat((p0, mat_to_quat(Y @ m.root_R[f0])), -1).expand(len(f), -1)
            d = dict(ref_root_p=m.root_p.numpy(), ref_root_R=m.root_R.numpy(),
                     start_frame=np.array([f0]), yaw=np.array([yaw]), init_root=init[:1].numpy())
            ghost = torch.tensor(np.stack([renderer.ghost_pose(d, int(i), 0)[0] for i in f]), dtype=torch.float32)
            torch.testing.assert_close(ideal_p, ghost, atol=4e-6, rtol=1e-6)
            for offset in (torch.zeros(3), torch.tensor([.7, -.4, .2])):
                S = (ideal_p+offset, R, mv3(Y, m.root_v[f]), mv3(Y, m.root_w[f]), m.q[f], m.qd[f])
                a = torch.zeros(len(f), 1)
                _, terms = mdp.reward(S, f, a, a, terms=True, init_root=init,
                                      init_f=torch.full_like(f, f0))
                world_error = ((S[0] - ghost)**2).sum(-1)
                torch.testing.assert_close(terms['e_root_pos'], world_error, atol=6e-6, rtol=2e-5)
                if not offset.any():
                    self.assertLess(terms['e_root_pos'].max().item(), 1e-9)

    def test_renderer_selects_all_episodes_including_repeated_motions(self):
        import importlib.util
        from pathlib import Path
        import numpy as np
        spec = importlib.util.spec_from_file_location('render_ran', Path(__file__).with_name('render_ran.py'))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        d = dict(names=['a', 'b'], root_p=np.zeros((5, 4, 3)),
                 rollout_clip=np.array([0, 1, 0, 0]), rollout_repeat=np.array([0, 0, 1, 2]))
        self.assertEqual(renderer.select_rollouts(d), [0, 1, 2, 3])
        self.assertEqual(renderer.select_rollouts(d, indices=[3, 0]), [3, 0])
        self.assertEqual(renderer.select_rollouts(d, clips=['a']), [0, 2, 3])
        self.assertEqual(renderer.select_rollouts(d, repeat=0), [0, 1])
        for kwargs in (dict(indices=[4]), dict(indices=[-1]), dict(repeat=3),
                       dict(indices=[0], clips=['a'])):
            with self.assertRaises(ValueError):
                renderer.select_rollouts(d, **kwargs)

    def test_ghost_rotates_about_sampled_frame_without_modifying_expert(self):
        import importlib.util
        from pathlib import Path
        import numpy as np
        spec = importlib.util.spec_from_file_location('render_ran', Path(__file__).with_name('render_ran.py'))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        d = dict(ref_root_p=self.source.root_p.numpy().copy(), ref_root_R=self.source.root_R.numpy().copy(),
                 start_frame=np.array([2, 5]), yaw=np.array([1.2, -1.4]))
        original = d['ref_root_p'].copy()
        for c, start in enumerate(d['start_frame']):
            p0, R0 = renderer.ghost_pose(d, int(start), c)
            np.testing.assert_allclose(p0, original[start])
            yaw = d['yaw'][c]
            Y = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            p, R = renderer.ghost_pose(d, int(start+1), c)
            np.testing.assert_allclose(p, original[start] + Y @ (original[start+1] - original[start]))
            np.testing.assert_allclose(R, Y @ d['ref_root_R'][start+1])
        np.testing.assert_array_equal(d['ref_root_p'], original)
        del d['start_frame']  # legacy recordings are already transformed
        p, R = renderer.ghost_pose(d, 2, 0)
        np.testing.assert_array_equal(p, original[2])


if __name__ == '__main__':
    unittest.main()
