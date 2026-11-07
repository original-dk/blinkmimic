"""CPU checks for packed NPZ motions, quaternion conventions and recursive mixed motion sets.

Run: python tools/test_motion_npz.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from blinkmimic.g1.g1_constants import load_lafan1_csv
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import CONTROL_DT, MotionLib, load_npz
from blinkmimic.rotations import mat_to_quat, quat_to_mat
import test_body_reference as bodies


class MotionNpzTest(unittest.TestCase):
    def setUp(self):
        fixture = bodies.BodyReferenceTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.kin = Path(fixture.tmp.name), fixture.kin
        m = fixture.m
        self.raw = np.concatenate((m.root_p.numpy(), mat_to_quat(m.root_R).numpy(), m.q.numpy()), -1)
        self.robot = SimpleNamespace(motion_glob='*.csv', load_motion=load_lafan1_csv)

    def write(self, name, order='wxyz', fps=30):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.raw.copy()
        if order == 'xyzw':
            raw[:, 3:7] = self.raw[:, [4, 5, 6, 3]]
        if path.suffix == '.npz':
            np.savez(path, qpos=raw, fps=np.float32(fps))
        else:
            np.savetxt(path, raw, delimiter=',')
        return str(path)

    def test_packed_fields_and_explicit_quaternion_orders(self):
        for order in ('wxyz', 'xyzw'):
            path = self.write(order+'.npz', order, fps=50)
            p, quat, joints, fps = load_npz(path, order)
            np.testing.assert_array_equal(p, self.raw[:, :3])
            np.testing.assert_array_equal(quat, self.raw[:, 3:7])
            np.testing.assert_array_equal(joints, self.raw[:, 7:])
            self.assertEqual(fps, 50.)

    def test_npz_and_csv_have_identical_states_fk_goals_and_rewards(self):
        csv = self.write('original.csv', 'xyzw')
        baseline = MotionLib([csv], self.kin, 'cpu', load_lafan1_csv)
        for order in ('wxyz', 'xyzw'):
            path = self.write(order+'.npz', order)
            motion = MotionLib([path], self.kin, 'cpu', load_lafan1_csv, order)
            for attr in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd', 'body_p', 'body_R', 'body_v', 'body_w',
                         'clip_start', 'clip_end', 'clip_id', 'frame_bin'):
                torch.testing.assert_close(getattr(motion, attr), getattr(baseline, attr), atol=0, rtol=0)
            frames = torch.arange(motion.num_frames)
            torch.testing.assert_close(motion.goal(frames, 10), baseline.goal(frames, 10), atol=0, rtol=0)
            mdp = MDP(self.kin, motion, torch.zeros(3), 4)
            S = motion.ref_state(frames)
            init = torch.cat((S[0], mat_to_quat(S[1])), -1)
            reward, terms = mdp.reward(S, frames, torch.zeros_like(S[4]), torch.zeros_like(S[4]), terms=True,
                                       init_root=init, init_f=frames)
            original = MDP(self.kin, baseline, torch.zeros(3), 4).reward(
                baseline.ref_state(frames), frames, torch.zeros_like(S[4]), torch.zeros_like(S[4]),
                init_root=init, init_f=frames)
            torch.testing.assert_close(reward, original, atol=0, rtol=0)
            torch.testing.assert_close(terms['track'], torch.full_like(reward, MAX_STEP_REWARD), atol=1e-6, rtol=0)

    def test_recursive_mixed_files_keep_unique_relative_names(self):
        self.write('a/walk.npz')
        self.write('b/deep/walk.csv', 'xyzw')
        self.write('root.csv', 'xyzw')
        (self.root/'ignored.txt').write_text('not a motion')
        motion = MotionLib.from_dir(str(self.root), self.robot, self.kin, 'cpu')
        self.assertEqual(motion.names, ['a/walk', 'b/deep/walk', 'root'])
        self.assertEqual(len(motion.clip_len), 3)
        n = int(motion.clip_len[0])
        for attr in ('root_p', 'root_R', 'q', 'root_v', 'root_w', 'body_p'):
            data = getattr(motion, attr)
            torch.testing.assert_close(data[:n], data[n:2*n], atol=0, rtol=0)

    def test_from_dir_forwards_xyzw_and_reads_source_fps(self):
        self.raw[:, 0] = np.arange(len(self.raw)) / 60
        self.write('nested/motion.npz', 'xyzw', fps=60)
        motion = MotionLib.from_dir(str(self.root), self.robot, self.kin, 'cpu', 'xyzw')
        self.assertEqual(motion.num_frames, int(np.floor((len(self.raw)-1)/60/CONTROL_DT))+1)
        torch.testing.assert_close(motion.root_v[:, 0], torch.ones(motion.num_frames), atol=2e-6, rtol=0)
        torch.testing.assert_close(motion.root_R[0], quat_to_mat(torch.from_numpy(self.raw[:1, 3:7]))[0])

    def test_empty_directory_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'No motions'):
            MotionLib.from_dir(str(self.root/'empty'), self.robot, self.kin, 'cpu')

    def test_training_cli_routes_npz_order_and_path_to_the_real_loader(self):
        from unittest.mock import patch
        path = self.write('nested/walk.npz', 'xyzw', fps=50)
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        text = source.read_text()
        ns = {'__file__': str(source)}
        with patch('sys.argv', ['train', '--motion_dir', str(self.root), '--motion_quat_order', 'xyzw']):
            exec(compile(text.split('\nimport torch\n', 1)[0], str(source), 'exec'), ns)
        stmt = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'motion' for t in n.targets))
        ns.update(MotionLib=MotionLib, robot=self.robot, kin=self.kin, dev='cpu')
        exec(compile(ast.Module(body=[stmt], type_ignores=[]), str(source), 'exec'), ns)
        actual = ns['motion']
        expected = MotionLib([path], self.kin, 'cpu', load_lafan1_csv, 'xyzw')
        self.assertEqual(actual.names, ['nested/walk'])
        torch.testing.assert_close(actual.root_R, expected.root_R, atol=0, rtol=0)
        self.assertEqual(actual.num_frames, len(self.raw))


if __name__ == '__main__':
    unittest.main()
