"""CPU checks for randomized evaluation references, observations and recorded ghost poses.

Run: python tools/test_random_yaw_eval.py
"""

import math
import unittest

import torch

from blinkmimic.evaluateran import random_yaw_motion
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import mat_to_quat, mv3, so3_exp
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

    def test_seed_range_and_source_unchanged(self):
        before = {k: v.clone() for k, v in vars(self.source).items() if isinstance(v, torch.Tensor)}
        torch.manual_seed(3)
        a, yaw = random_yaw_motion(self.source, 3, math.pi / 2)
        torch.manual_seed(3)
        b, yaw_b = random_yaw_motion(self.source, 3, math.pi / 2)
        torch.testing.assert_close(yaw, yaw_b)
        torch.testing.assert_close(a.root_p, b.root_p)
        self.assertTrue((yaw.abs() <= math.pi / 2).all())
        self.assertEqual(len(torch.unique(yaw)), 6)
        for k, v in before.items():
            torch.testing.assert_close(getattr(self.source, k), v, atol=0, rtol=0)
        self.assertIsNot(a, self.source)

    def test_repeats_metadata_and_zero_yaw(self):
        m, _ = random_yaw_motion(self.source, 3, 0)
        torch.testing.assert_close(m.clip_start, torch.tensor([0, 4, 8, 12, 16, 20]))
        torch.testing.assert_close(m.clip_end, m.clip_start + 3)
        torch.testing.assert_close(m.bin_start[m.frame_bin], m.clip_start[m.clip_id])
        for c in range(6):
            f = torch.tensor([m.clip_start[c]])
            torch.testing.assert_close(m.future_frames(f, 8).max(), m.clip_end[c])
        for k in ('root_p', 'root_R', 'body_p', 'body_R', 'root_v', 'root_w', 'body_v', 'body_w'):
            x = getattr(self.source, k)
            torch.testing.assert_close(getattr(m, k), x.repeat(3, *([1] * (x.ndim - 1))))

    def test_positions_rotations_and_velocities_follow_rollout_yaw(self):
        m, yaw = random_yaw_motion(self.source, 3, math.pi)
        for c in range(6):
            original = c % 2
            s, s0 = int(m.clip_start[c]), int(self.source.clip_start[original])
            Y = so3_exp(torch.tensor([0., 0., yaw[c]]))
            origin = self.source.root_p[s0]
            for t in range(4):
                torch.testing.assert_close(m.root_p[s+t], origin + Y @ (self.source.root_p[s0+t] - origin))
                torch.testing.assert_close(m.root_R[s+t], Y @ self.source.root_R[s0+t])
                torch.testing.assert_close(m.body_p[s+t, 0], m.root_p[s+t])
                torch.testing.assert_close(m.body_R[s+t, 0], m.root_R[s+t])
                torch.testing.assert_close(m.root_v[s+t], Y @ self.source.root_v[s0+t])
                torch.testing.assert_close(m.root_w[s+t], Y @ self.source.root_w[s0+t])
            torch.testing.assert_close(m.root_p[s], origin)

    def test_policy_observations_goals_rewards_and_termination_are_aligned(self):
        m, _ = random_yaw_motion(self.source, 3, math.pi)
        base = MDP(RootKinematics(), self.source, torch.zeros(1), 1)
        aligned = MDP(RootKinematics(), m, torch.zeros(1), 1)
        f = m.clip_start + 1
        f_base = self.source.clip_start.repeat(3) + 1
        S, ref = m.ref_state(f), self.source.ref_state(f_base)
        a, xi = torch.zeros(6, 1), torch.zeros(6, 1)
        torch.testing.assert_close(aligned.obs(S, a, xi, f), base.obs(ref, a, xi, f_base), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(m.goal(f, 5), self.source.goal(f_base, 5), atol=2e-6, rtol=1e-5)
        init = torch.cat((m.root_p[m.clip_start], mat_to_quat(m.root_R[m.clip_start])), -1)
        init_base = torch.cat((self.source.root_p[self.source.clip_start],
                               mat_to_quat(self.source.root_R[self.source.clip_start])), -1).repeat(3, 1)
        torch.testing.assert_close(aligned.reward(S, f, a, a, init_root=init, init_f=m.clip_start),
                                   base.reward(ref, f_base, a, a, init_root=init_base,
                                               init_f=self.source.clip_start.repeat(3)))
        torch.testing.assert_close(aligned.termination(S, f), base.termination(ref, f_base))
        # Absolute reference input would command a turn back; aligned input removes that offset.
        self.assertGreater((base.obs(S, a, xi, f_base) - aligned.obs(S, a, xi, f)).abs().max().item(), .1)


if __name__ == '__main__':
    unittest.main()
