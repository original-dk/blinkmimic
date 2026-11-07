"""Compact motion storage, on-demand reference FK, precision and bounded training entry checks (CPU).

Run: python tools/test_compact_motion.py
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from blinkmimic.g1.g1_constants import load_lafan1_csv
from blinkmimic.env import TrackingSim
from blinkmimic.mdp import MDP
from blinkmimic.motion import BIN_FRAMES, MotionLib
from blinkmimic.rotations import mat_to_quat, so3_exp, so3_log
import test_body_reference as bodies


class CompactMotionTest(unittest.TestCase):
    def setUp(self):
        fixture = bodies.BodyReferenceTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.kin = Path(fixture.tmp.name), fixture.kin
        m = fixture.m
        self.raw = np.concatenate((m.root_p.numpy(), mat_to_quat(m.root_R).numpy(), m.q.numpy()), -1)

    def library(self, lengths=(8,), dtype=torch.float16, joints=3):
        files = []
        for i, n in enumerate(lengths):
            path = self.root/f'clip{i}.npz'
            raw = np.resize(self.raw, (n, 10)).copy()
            raw[:, :3] = self.raw[0, :3] + np.arange(n)[:, None] * .002
            raw[:, 3:7] = self.raw[0, 3:7]
            if joints != 3:
                raw = np.concatenate((raw[:, :7], np.zeros((n, joints), dtype=np.float32)), -1)
            np.savez(path, qpos=raw, fps=np.float32(50))
            files.append(str(path))
        return MotionLib(files, self.kin, 'cpu', load_lafan1_csv, storage_dtype=dtype)

    def test_no_constructor_fk_or_full_frame_index_arrays(self):
        with patch.object(self.kin, 'forward', side_effect=AssertionError('eager reference FK')):
            m = self.library((103, 59), joints=29)
        self.assertEqual(m.root_p.dtype, torch.float32)
        self.assertEqual(m.rest.dtype, torch.float16)
        for key in ('root_R', 'body_p', 'body_R', 'body_v', 'body_w', 'clip_id', 'frame_bin', 'root_v_local'):
            self.assertNotIn(key, m.__dict__)
        frames_bytes = m.root_p.numel()*4 + m.rest.numel()*2
        self.assertEqual(frames_bytes, m.num_frames*148)
        for view in (m.root_quat, m.root_v, m.root_w, m.q, m.qd):
            self.assertEqual(view.untyped_storage().data_ptr(), m.rest.untyped_storage().data_ptr())
        metadata = sum(x.numel()*x.element_size() for x in
                       (m.clip_start, m.clip_end, m.clip_len, m.clip_bin0, m.bin_clip, m.bin_start))
        self.assertEqual(m.storage_bytes, frames_bytes+metadata)

    def test_lookup_matches_dense_indices_at_clip_and_bin_boundaries(self):
        m = self.library((2, 49, 50, 51, 101))
        frames = torch.arange(m.num_frames)
        dense_clips, dense_bins = [], []
        for c, n in enumerate(m.clip_len.tolist()):
            dense_clips.extend([c]*n)
            dense_bins.extend((m.clip_bin0[c]+torch.arange(n)//BIN_FRAMES).tolist())
        torch.testing.assert_close(m.clip_of(frames), torch.tensor(dense_clips))
        torch.testing.assert_close(m.bin_of(frames), torch.tensor(dense_bins))
        grid = frames[:12].reshape(3, 4)
        torch.testing.assert_close(m.bin_of(grid), torch.tensor(dense_bins)[:12].reshape(3, 4))
        future = m.future_frames(frames, 10)
        self.assertTrue((future <= m.clip_end[m.clip_of(frames), None]).all())

    def test_initialization_temporaries_are_bounded_by_one_clip(self):
        original, sizes = torch.cat, []
        def counted(*args, **kwargs):
            out = original(*args, **kwargs)
            sizes.append(out.numel())
            return out
        with patch('torch.cat', counted):
            m = self.library((8,)*20)
        self.assertEqual(m.num_frames, 160)
        self.assertLessEqual(max(sizes), 8*(10+2*3))

    def test_reset_never_samples_a_terminal_only_bin_as_an_initial_frame(self):
        m = self.library((51, 101))
        env = TrackingSim.__new__(TrackingSim)
        env.motion, env.device = m, 'cpu'
        env.bin_G, env.bin_w_avg, env.G_max = torch.zeros(m.num_bins), torch.ones(m.num_bins), 1.
        env.fail_reset, env.focus_static, env.focus_q = 0., False, None
        # Force the bin containing only the final frame of the last clip.
        with patch('torch.multinomial', return_value=torch.full((32,), m.num_bins-1)):
            f, bins = env.sample_frames(32)
        self.assertTrue((f == m.clip_end[-1]-1).all())
        self.assertTrue((bins == m.num_bins-1).all())

    def test_actual_imagination_step_never_reads_past_a_short_clip(self):
        m = self.library((2, 8))
        mdp = MDP(self.kin, m, torch.zeros(3), 4)
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_imag_step')
        ns = dict(torch=torch, mdp=mdp, motion=m, K=10,
                  pi=lambda hist, goal: torch.zeros(1, 3), wm=SimpleNamespace(step=lambda S, *args: S))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        f = m.clip_end[:1].clone()
        S, hist, prev, xi = m.ref_state(f), torch.zeros(1, 2, mdp.obs_dim), torch.zeros(1, 3), torch.zeros(1, 4)
        for _ in range(5):
            S, prev, hist, f, g = ns['_imag_step'](S, hist, prev, f, xi, xi, m.goal(f, 10))
            self.assertEqual(f.item(), m.clip_end[0].item())
            torch.testing.assert_close(g, m.goal(f, 10))

    def test_packed_precision_and_fp32_queries(self):
        path = self.root/'motion.npz'
        raw = self.raw.copy()
        # Smooth physical motion; root position must retain submillimeter detail far from the origin.
        raw[:, :3] = 100 + self.raw[0, :3] + np.arange(len(raw))[:, None]*.002
        t = torch.arange(len(raw))*.02
        raw[:, 3:7] = mat_to_quat(so3_exp(torch.stack((t*.1, t*.2, t*.3), -1))).numpy()
        raw[:, 7:] = (t[:, None]*torch.tensor([[.1, .2, .3]])).sin().numpy()
        np.savez(path, qpos=raw, fps=50)
        a = MotionLib([str(path)], self.kin, 'cpu', load_lafan1_csv)
        b = MotionLib([str(path)], self.kin, 'cpu', load_lafan1_csv, storage_dtype=torch.float32)
        f = torch.arange(a.num_frames)
        A, B = a.ref_state(f), b.ref_state(f)
        torch.testing.assert_close(A[0], B[0], atol=0, rtol=0)
        self.assertTrue(all(x.dtype == torch.float32 for x in A))
        self.assertLess(so3_log(A[1].transpose(-1, -2)@B[1]).norm(dim=-1).max().item(), 1e-3)
        for x, y in zip(A[2:], B[2:]):
            torch.testing.assert_close(x, y, atol=2e-3, rtol=6e-4)
        torch.testing.assert_close(a.goal(f, 10), b.goal(f, 10), atol=3e-3, rtol=1e-3)

    def test_reference_contact_uses_only_compact_boolean_cache(self):
        m = self.library((103, 59))
        f = torch.arange(m.num_frames)
        bp, _, bv, _ = m.body_state(f)
        expected = torch.empty(m.num_frames, 2, dtype=torch.bool)
        for s, n in zip(m.clip_start.tolist(), m.clip_len.tolist()):
            z, speed = bp[s:s+n, [0, 1], 2], bv[s:s+n, [0, 1], :2].norm(dim=-1)
            expected[s:s+n] = (z < z.quantile(.05, dim=0)+.015) & (speed < .5)
        before = m.storage_bytes
        m.set_ref_contact([0, 1])
        torch.testing.assert_close(m.ref_contact, expected)
        self.assertEqual(m.ref_contact.dtype, torch.bool)
        self.assertEqual(m.storage_bytes-before, 2*m.num_frames)
        self.assertNotIn('body_p', m.__dict__)

    def test_compiled_reward_goals_critic_and_reference_gradients(self):
        m = self.library((8, 8))
        mdp = MDP(self.kin, m, torch.zeros(3), 4)
        mdp.feet = [0]
        m.set_ref_contact([0])
        f = torch.tensor([2, 10])
        S = list(m.ref_state(f))
        init = torch.cat((S[0], mat_to_quat(S[1])), -1)
        S[0] = (S[0]+.1).requires_grad_()
        S[4] = (S[4]+.1).requires_grad_()
        a = torch.full((2, 3), .2, requires_grad=True)
        def fn(p, q, action):
            state = (p, *S[1:4], q, S[5])
            return (mdp.reward(state, f, action, action*0, init_root=init, init_f=f, w_contact=.25),
                    m.goal(f, 10), mdp.critic_features(state, f, torch.zeros(2, 4), init, f))
        inputs = (S[0], S[4], a)
        eager = fn(*inputs)
        compiled = torch.compile(fn, backend='aot_eager', fullgraph=True)(*inputs)
        torch.testing.assert_close(compiled, eager)
        for x in torch.autograd.grad(compiled[0].sum()+compiled[2].sum(), inputs):
            self.assertTrue(torch.isfinite(x).all())
        self.assertIsNone(m.rest.grad)

    def test_recording_reference_arrays_are_float32_on_cpu(self):
        m = self.library()
        arrays = m.reference_arrays()
        expected = m.ref_state(torch.arange(m.num_frames))
        for key, value in zip(('ref_root_p', 'ref_root_R', 'ref_q'), (expected[0], expected[1], expected[4])):
            self.assertEqual(arrays[key].dtype, np.float32)
            np.testing.assert_array_equal(arrays[key], value.numpy())

    def test_invalid_motion_does_not_silently_drop_a_clip(self):
        path = self.root/'bad.npz'
        for n, fps in ((1, 50), (8, 0), (8, float('nan'))):
            np.savez(path, qpos=self.raw[:n], fps=fps)
            with self.assertRaisesRegex(ValueError, 'two frames'):
                MotionLib([str(path)], self.kin, 'cpu', load_lafan1_csv)

    def test_32gb_entry_defaults_and_override_reach_actual_parser(self):
        repo = Path(__file__).parents[1]
        shim = self.root/'parse-python'
        shim.write_text(f'#!{sys.executable}\nimport sys, json\nfrom pathlib import Path\n'
                        f'p = Path({str(repo/"src/blinkmimic/train.py")!r})\n'
                        'sys.argv = ["train", *sys.argv[4:]]\nns = {"__file__": str(p)}\n'
                        'exec(p.read_text().split("\\nimport torch\\n", 1)[0], ns)\nprint(json.dumps(vars(ns["args"])))\n')
        shim.chmod(0o755)
        env = {**os.environ, 'PYTHON': str(shim)}
        cmd = ['bash', str(repo/'scripts/train_bones32.sh'), str(self.root/'logs'), '--motion_dir', str(self.root)]
        out = subprocess.run(cmd, env=env, text=True, capture_output=True)
        self.assertEqual(out.returncode, 0, out.stderr+out.stdout)
        args = json.loads(out.stdout)
        for key, value in dict(num_envs=2048, buffer_steps=500, pi_batch=1024, wm_batch=1024,
                               eprobe_share=0, eprobe_async=0, max_episode_steps=200).items():
            self.assertEqual(args[key], value)
        override = subprocess.run(cmd+['--num_envs', '1024'], env=env, text=True, capture_output=True)
        self.assertEqual(json.loads(override.stdout)['num_envs'], 1024)
        failed = subprocess.run(cmd+['--invalid-option'], env=env, text=True, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)  # train.sh must propagate failure through tee

    def test_probe_initialization_and_sweep_cap_parallel_worlds(self):
        m = self.library((103, 59))
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        tree = ast.parse(source.read_text())
        init = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'EPROBE')
        burst = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eprobe_burst')
        args = SimpleNamespace(eprobe_tile=50, eprobe_envs=2, eprobe_async=0, eprobe_graph=False,
                               eprobe_lead_min=2, eprobe_lead_max=3)
        import time
        factory = Mock(return_value=SimpleNamespace(done_eval=torch.zeros(2, dtype=torch.bool),
                                                    failed_eval=torch.zeros(2, dtype=torch.bool), reset=Mock()))
        ns = dict(torch=torch, time=time, args=args, motion=m, EPROBE=True, dev='cpu',
                  pi_avg=SimpleNamespace(module=Mock()), robot=Mock(), mdp=Mock(),
                  TrackingSim=factory, make_sim=Mock(return_value=None),
                  eprobe_sweep_eager=Mock(return_value=torch.full((2,), -1)), eprobe_finish=Mock())
        exec(compile(ast.Module(body=[init, burst], type_ignores=[]), str(source), 'exec'), ns)
        self.assertEqual(ns['P_EP'], 2)
        self.assertGreater(len(ns['EP_TILE']), 2)
        self.assertEqual(factory.call_args.args[1], 2)
        ns['eprobe_burst'](1)
        self.assertEqual(factory.return_value.reset.call_args.args[1].shape, (2,))
        self.assertEqual(ns['eprobe_sweep_eager'].call_args.args[1].shape, (2,))

    def test_memory_check_reports_early_training_failure_without_events(self):
        import check_bones_memory as check
        log = self.root/'failed-check'
        process = Mock(returncode=2)
        process.poll.return_value = 2
        with patch('sys.argv', ['check', '--motion_dir', str(self.root), '--log_dir', str(log)]), \
             patch.object(check, 'gpu_memory', return_value=(1024, 32768)), \
             patch.object(check.subprocess, 'Popen', return_value=process), \
             patch('builtins.print'), self.assertRaises(SystemExit) as ended:
            check.main()
        self.assertEqual(ended.exception.code, 1)
        report = json.loads((log/'memory_check.json').read_text())
        self.assertFalse(report['passed'])
        self.assertEqual(report['returncode'], 2)
        self.assertEqual(report['last_logged_iteration'], 0)


if __name__ == '__main__':
    unittest.main()
