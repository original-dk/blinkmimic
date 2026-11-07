"""FP32 host/GPU cache equivalence, eviction, replay IDs and compiled gradient checks.

PYTHONPATH=src:tools python -m unittest test_motion_cache
CUDA-only checks run automatically when a GPU is available.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.env import TrackingSim
from blinkmimic.g1.g1_constants import load_lafan1_csv
from blinkmimic.mdp import MDP, flatten_state
from blinkmimic.mdp import MAX_STEP_REWARD, unflatten_state
from blinkmimic.motion import MotionLib
from blinkmimic.motion_cache import CachedMotionLib
from blinkmimic.motion_storage import GIB, MappedStates, check_disk, check_memory, memory_budget
from blinkmimic.networks import Critic, Policy, WorldModel
from blinkmimic.objective import actor_objective
from blinkmimic.rotations import mat_to_quat, so3_exp
import test_body_reference as bodies


class MotionCacheTest(unittest.TestCase):
    def setUp(self):
        fixture = bodies.BodyReferenceTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.kin = Path(fixture.tmp.name), fixture.kin
        self.files = []
        for i, n in enumerate((51, 37)):
            path = self.root/f'clip{i}.npz'
            t = torch.arange(n).float()
            p = torch.stack((t*.012, (t*.1).sin()*.2, .8+t*.001), -1)
            R = so3_exp(torch.stack(((t*.1).sin()*.2, (t*.2).cos()*.1, t*.03), -1))
            q = torch.stack(((t*.1).sin()*.2, (t*.2).cos()*.1, t*.002), -1)
            np.savez(path, qpos=torch.cat((p, mat_to_quat(R), q), -1).numpy(), fps=np.float32(50))
            self.files.append(str(path))
        self.dense = MotionLib(self.files, self.kin, 'cpu', load_lafan1_csv)
        # 12 small pages force genuine replacement while keeping a whole test batch resident.
        width = 18 + 2*3 + 4*18 + 6
        self.page_bytes = 4*width*4
        self.cache = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv,
                                     cache_gib=12*self.page_bytes/2**30, page_size=4)

    def test_all_fp32_fields_and_commands_match_resident_library(self):
        a, b = self.cache, self.dense
        for name in ('clip_len', 'clip_start', 'clip_end', 'clip_id', 'bin_clip', 'bin_start', 'frame_bin'):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        for begin in range(0, a.num_frames, 8):
            f = torch.arange(begin, min(begin+8, a.num_frames))
            a.prepare(f, ahead=3)
            for name in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd', 'body_p', 'body_R', 'body_v', 'body_w',
                         'root_v_local', 'root_w_local', 'root_gz'):
                field = getattr(a, name)
                self.assertEqual(field.dtype, torch.float32)
                torch.testing.assert_close(field[f], getattr(b, name)[f], rtol=0, atol=0)
            torch.testing.assert_close(a.goal(f, 3), b.goal(f, 3), rtol=0, atol=0)
        self.assertGreater(a.loaded_pages, a.capacity)
        self.assertLess(a.data.shape[0], a.num_frames)

    def test_reference_queries_never_run_fk(self):
        f = torch.tensor([2, 54])
        self.cache.prepare(f, ahead=3)
        self.kin.forward = Mock(side_effect=AssertionError('reference FK during training'))
        self.cache.goal(f, 3)
        self.cache.ref_state(f)
        self.cache.body_R[f]

    def test_cache_hit_does_not_copy_pages_again(self):
        self.cache.prepare(torch.tensor([2, 54]), ahead=4)
        before = self.cache.loaded_pages
        self.cache.prepare(torch.tensor([2, 54]), ahead=4)
        self.assertEqual(self.cache.loaded_pages, before)

    def test_only_raw_fp32_state_is_stored_on_the_host(self):
        self.assertEqual(self.cache.host.shape, (88, 18+2*3))
        self.assertEqual(self.cache.host_bytes, 88*(18+2*3)*4)
        for name in ('body_p', 'body_R', 'body_v', 'body_w', 'root_v_local', 'root_w_local'):
            self.assertIsNone(getattr(self.cache, name).host)
        self.kin.forward = Mock(wraps=self.kin.forward)
        self.cache.prepare(torch.tensor([2, 54]))
        calls = self.kin.forward.call_count
        self.assertGreater(calls, 0)
        self.cache.prepare(torch.tensor([2, 54]))
        self.assertEqual(self.kin.forward.call_count, calls)

    def test_ssd_mapping_reuses_completed_states_without_reloading_sources(self):
        directory = self.root/'ssd-cache'
        a = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        self.assertIsInstance(a.storage, MappedStates)
        self.assertEqual(a.storage.path.stat().st_size, a.num_frames*(18+2*3)*4)
        with patch('blinkmimic.motion_cache.load_npz', side_effect=AssertionError('reloading completed cache')):
            b = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        f = torch.tensor([2, 54])
        b.prepare(f, ahead=3)
        torch.testing.assert_close(b.goal(f, 3), self.dense.goal(f, 3), atol=0, rtol=0)
        # Dropping mapped pages must preserve file contents and all derived-cache queries.
        b.storage.release(force=True)
        torch.testing.assert_close(b.root_p.cpu(), self.dense.root_p, atol=0, rtol=0)

    def test_changed_source_does_not_reuse_stale_disk_states(self):
        directory = self.root/'ssd-change'
        a = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        with np.load(self.files[0]) as d:
            raw = d['qpos'].copy()
        raw[:, 0] += 5
        np.savez(self.files[0], qpos=raw, fps=np.float32(50))
        b = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        self.assertNotEqual(a.storage.path, b.storage.path)
        torch.testing.assert_close(b.root_p.cpu()[0], a.root_p.cpu()[0]+torch.tensor([5., 0., 0.]), atol=1e-6, rtol=0)

    def test_interrupted_build_restarts_and_active_builder_is_not_overwritten(self):
        import fcntl
        directory = self.root/'ssd-incomplete'
        a = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        marker = a.storage.path.parent/'complete.json'
        marker.unlink()
        fcntl.flock(a._lock, fcntl.LOCK_EX)
        with self.assertRaisesRegex(RuntimeError, 'being built'):
            CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        a._lock.close()  # simulate release of the crashed builder's lock
        b = CachedMotionLib(self.files, self.kin, 'cpu', load_lafan1_csv, cache_gib=.001, cache_dir=directory)
        self.assertTrue(marker.exists())
        torch.testing.assert_close(b.root_p.cpu(), self.dense.root_p, atol=0, rtol=0)

    def test_cgroup_limit_and_ancestor_limits_override_machine_ram(self):
        meminfo, cg, membership = self.root/'meminfo', self.root/'cgroup', self.root/'membership'
        meminfo.write_text(f'MemTotal: {96*GIB//1024} kB\nMemAvailable: {80*GIB//1024} kB\n')
        cg.mkdir()
        (cg/'memory.max').write_text('max')
        (cg/'memory.current').write_text('0')
        child = cg/'user'/'worker'; child.mkdir(parents=True)
        (child/'memory.max').write_text(str(16*GIB))
        (child/'memory.current').write_text(str(14*GIB))
        membership.write_text('0::/user/worker\n')
        self.assertEqual(memory_budget(meminfo, cg, membership), (2*GIB, 16*GIB))
        (cg/'user'/'memory.max').write_text(str(12*GIB))
        (cg/'user'/'memory.current').write_text(str(11*GIB))
        self.assertEqual(memory_budget(meminfo, cg, membership), (GIB, 12*GIB))

    def test_insufficient_ram_exits_before_allocating_large_tensors(self):
        class LargeMotion:
            def __len__(self):
                return 51_888_186
        loader = lambda path: (LargeMotion(), None, np.zeros((1, 29)), 50.)
        with patch('blinkmimic.motion_storage.memory_budget', return_value=(GIB, 96*GIB)):
            with patch('blinkmimic.motion_cache.torch.empty', side_effect=AssertionError('allocated before preflight')):
                with self.assertRaisesRegex(MemoryError, 'RAM/cgroup'):
                    CachedMotionLib(['unused.csv'], self.kin, 'cpu', loader)

    def test_insufficient_disk_and_ram_filesystems_are_rejected(self):
        path = self.root/'ssd-full'/'states.fp32'
        with patch('blinkmimic.motion_storage.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'free disk'):
                MappedStates(path, 10, 24, create=True)
        self.assertFalse(path.exists())
        mounts = self.root/'mountinfo'
        mounts.write_text(f'1 0 0:1 / / rw - ext4 /dev/root rw\n2 1 0:2 / {self.root} rw - tmpfs tmpfs rw\n')
        with self.assertRaisesRegex(ValueError, 'RAM filesystem'):
            check_disk(self.root/'ram-cache', 100, mountinfo=mounts)

    def test_mapped_mode_does_not_require_full_reference_file_in_available_ram(self):
        with patch('blinkmimic.motion_storage.memory_budget', return_value=(8*GIB, 16*GIB)):
            check_memory(51_888_186, 76, mapped=True)
            with self.assertRaises(MemoryError):
                check_memory(51_888_186, 76, mapped=False)

    def test_one_clip_keeps_integer_metadata_and_explicit_xyzw_order(self):
        path = self.root/'single_xyzw.npz'
        with np.load(self.files[0]) as d:
            raw = d['qpos'].copy()
        raw[:, 3:7] = raw[:, [4, 5, 6, 3]]
        np.savez(path, qpos=raw, fps=np.float32(50))
        robot = SimpleNamespace(motion_glob='*.csv', load_motion=load_lafan1_csv)
        sub = self.root/'nested'; sub.mkdir()
        path.rename(sub/path.name)
        c = MotionLib.from_dir(str(sub), robot, self.kin, 'cpu', 'xyzw', cache_gib=.001)
        self.assertEqual(c.names, ['single_xyzw'])
        self.assertEqual(c.clip_start.dtype, torch.long)
        f = torch.tensor([0, 25, 50])
        c.prepare(f, ahead=3)
        torch.testing.assert_close(c.goal(f, 3), self.dense.goal(f, 3), rtol=0, atol=0)

    def test_eviction_reloads_replay_global_ids_and_reset_context(self):
        c, m = self.cache, self.dense
        f, init_f = torch.tensor([6, 57]), torch.tensor([1, 53])
        init = torch.cat((m.root_p[init_f], mat_to_quat(m.root_R[init_f])), -1)
        buffer = ReplayBuffer(2, 2, 'cpu', 3, 1)
        tr = {k: torch.zeros_like(v[0]) for k, v in buffer.data.items()}
        tr.update(s=flatten_state(m.ref_state(f)), f=f, init_root=init.clone(), init_f=init_f.clone())
        buffer.add(tr)
        c.prepare(torch.arange(32, 72))
        s, _ = buffer.starts_at(torch.zeros(2, dtype=torch.long), torch.arange(2), 1)
        c.prepare(s['f'], ahead=3, extra=s['init_f'])
        torch.testing.assert_close(c.root_p[s['f']], m.root_p[f], rtol=0, atol=0)
        torch.testing.assert_close(c.root_R[s['init_f']], m.root_R[init_f], rtol=0, atol=0)
        torch.testing.assert_close(s['init_root'], init)

    def test_oversized_batch_fails_before_evicting_anything(self):
        self.cache.prepare(torch.tensor([2, 54]))
        old = self.cache.page_table.clone()
        with self.assertRaisesRegex(RuntimeError, 'increase --motion_cache_gib'):
            self.cache.prepare(torch.arange(self.cache.num_frames))
        torch.testing.assert_close(self.cache.page_table, old)
        with self.assertRaises(ValueError):
            self.cache.prepare(torch.tensor([-1]))

    def test_missing_prefetch_fails_instead_of_returning_a_stale_slot(self):
        with self.assertRaises(IndexError):
            self.cache.root_p[torch.tensor([0])]

    def test_contact_cache_matches_original_and_survives_eviction(self):
        self.cache.prepare(torch.tensor([0]))
        self.dense.set_ref_contact([1, 3])
        self.cache.set_ref_contact([1, 3])
        torch.testing.assert_close(self.cache.ref_contact.cpu(), self.dense.ref_contact, rtol=0, atol=0)
        for f in (torch.arange(32), torch.arange(56, 88), torch.arange(16)):
            self.cache.prepare(f)
            torch.testing.assert_close(self.cache.ref_contact[f], self.dense.ref_contact[f], rtol=0, atol=0)

    def test_reward_critic_and_compiled_gradients_match_original(self):
        f, f0 = torch.tensor([6, 57]), torch.tensor([1, 53])
        self.cache.prepare(f, ahead=3, extra=f0)
        init = torch.cat((self.dense.root_p[f0], mat_to_quat(self.dense.root_R[f0])), -1)
        q = (self.dense.q[f]+.04).requires_grad_()
        p = (self.dense.root_p[f]+.03).requires_grad_()
        S = list(self.dense.ref_state(f)); S[0], S[4] = p, q
        a = torch.ones(2, 3, requires_grad=True)*.1
        mdps = [MDP(self.kin, m, torch.zeros(3), 1) for m in (self.dense, self.cache)]
        def fn(mdp, pp, qq, aa):
            state = (pp, *S[1:4], qq, S[5])
            reward = mdp.reward(state, f, aa, aa*0, init_root=init, init_f=f0)
            feat = mdp.critic_features(state, f, torch.zeros(2, 1), init, f0)
            return reward, feat
        expected = fn(mdps[0], p, q, a)
        actual = fn(mdps[1], p, q, a)
        compiled = torch.compile(lambda pp, qq, aa: fn(mdps[1], pp, qq, aa), backend='aot_eager', fullgraph=True)
        result = compiled(p, q, a)
        for values in (actual, result):
            for x, y in zip(values, expected):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        lhs = torch.autograd.grad(sum(x.square().sum() for x in expected), (p, q, a), retain_graph=True)
        rhs = torch.autograd.grad(sum(x.square().sum() for x in result), (p, q, a))
        for x, y in zip(lhs, rhs):
            self.assertTrue(torch.isfinite(y).all())
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)

    def test_policy_prefetch_refreshes_static_batch_without_freezing_sampling(self):
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_policy_batch')
        samples = [({'f': torch.tensor([6]), 'init_f': torch.tensor([1])}, {'s': torch.zeros(1, 2)}),
                   ({'f': torch.tensor([57]), 'init_f': torch.tensor([53])}, {'s': torch.ones(1, 2)})]
        ns = dict(POLICY_BATCH=None, motion=self.cache, HZ=2, K=3, args=SimpleNamespace(pi_batch=1),
                  sample_valid_starts=Mock(side_effect=samples))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        ns['prepare_policy_batch']()
        pointers = [t.data_ptr() for d in ns['POLICY_BATCH'] for t in d.values()]
        ns['prepare_policy_batch']()
        self.assertEqual(pointers, [t.data_ptr() for d in ns['POLICY_BATCH'] for t in d.values()])
        self.assertEqual(ns['sample_valid_starts'].call_count, 2)
        torch.testing.assert_close(self.cache.root_p[ns['POLICY_BATCH'][0]['f']], self.dense.root_p[torch.tensor([57])])
        self.assertEqual(ns['POLICY_BATCH'][1]['s'].sum().item(), 2)

    def test_actual_policy_update_matches_original_with_staged_references(self):
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        names = {'_imag_step', '_critic_input', '_reward_term', '_gae_targets', '_critic_loss',
                 'policy_update', 'prepare_policy_batch'}
        nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        outputs, parameters = [], []
        for m in (self.dense, self.cache):
            torch.manual_seed(112)
            f, f0 = torch.tensor([6, 57]), torch.tensor([1, 53])
            init = torch.cat((self.dense.root_p[f0], mat_to_quat(self.dense.root_R[f0])), -1)
            mdp = MDP(self.kin, m, torch.zeros(3), 4)
            mdp.feet = [1, 3]
            pi = Policy(mdp.obs_dim, m.goal_dim(2), 3, hist=2, enc_hidden=(8,), dec_hidden=(8,))
            wm = WorldModel(3, 4, hidden=(8,), structured=True)
            critic = Critic(mdp.critic_dim+m.goal_dim(2))
            s = dict(s=flatten_state(self.dense.ref_state(f)), f=f, init_f=f0, init_root=init,
                     a_prev=torch.zeros(2, 3), xi=torch.zeros(2, 4))
            h = {k: s[k][:, None].expand(2, 2, *s[k].shape[1:]) for k in ('s', 'f', 'xi', 'a_prev')}
            args = SimpleNamespace(pi_batch=2, gamma=.99, lam=.95, compile_more=False,
                                   pi_clip_value=0., no_sync=False, fused_adam=False, critic_epochs=1, critic_minibatches=1)
            ns = dict(torch=torch, nn=torch.nn, mdp=mdp, motion=m, pi=pi, wm=wm, critic=critic,
                      args=args, HZ=2, Hh=2, K=2, GRAPH=False, dev='cpu', MAX_STEP_REWARD=MAX_STEP_REWARD,
                      POLICY_BATCH=None, unflatten_state=unflatten_state, normalize_xi=lambda x: x,
                      sample_valid_starts=lambda n: (s, h), pipe=SimpleNamespace(stateful=False), pi_avg=None,
                      actor_objective=actor_objective, set_requires_grad=lambda model, flag: model.requires_grad_(flag),
                      opt_pi=torch.optim.Adam(pi.parameters(), lr=1e-4), opt_v=torch.optim.Adam(critic.parameters(), lr=1e-4))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
            ns.update(imag_step=ns['_imag_step'], reward_term=ns['_reward_term'], critic_input_c=ns['_critic_input'])
            if m.cached:
                ns['prepare_policy_batch']()
            torch.manual_seed(215)
            outputs.append(ns['policy_update']())
            parameters.append([p.detach().clone() for net in (pi, critic) for p in net.parameters()])
        for a, b in zip(outputs[0], outputs[1]):
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(parameters[0], parameters[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_actual_probe_limits_worlds_and_prefetches_the_sweep(self):
        import time
        source = Path(__file__).parents[1]/'src/blinkmimic/train.py'
        tree = ast.parse(source.read_text())
        init = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'EPROBE')
        burst = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eprobe_burst')
        args = SimpleNamespace(eprobe_tile=4, eprobe_envs=2, eprobe_async=0, eprobe_lead_min=2,
                               eprobe_lead_max=2, eprobe_graph=False)
        penv = SimpleNamespace(done_eval=torch.zeros(2, dtype=torch.bool), failed_eval=torch.zeros(2, dtype=torch.bool),
                               reset=Mock())
        factory = Mock(return_value=penv)
        ns = dict(torch=torch, time=time, args=args, motion=self.cache, EPROBE=True, dev='cpu', K=2,
                  pi_avg=SimpleNamespace(module=Mock()), robot=Mock(), mdp=Mock(),
                  TrackingSim=factory, make_sim=Mock(return_value=None),
                  eprobe_sweep_eager=Mock(return_value=torch.full((2,), -1)), eprobe_finish=Mock())
        exec(compile(ast.Module(body=[init, burst], type_ignores=[]), str(source), 'exec'), ns)
        self.assertEqual(ns['P_EP'], 2)
        self.assertGreater(len(ns['EP_TILE']), 2)
        self.assertEqual(factory.call_args.args[1], 2)
        ns['eprobe_burst'](1000)
        f0 = penv.reset.call_args.args[1]
        torch.testing.assert_close(self.cache.goal(f0, 2), self.dense.goal(f0, 2), rtol=0, atol=0)
        ns['eprobe_finish'].assert_called_once()

    def test_terminal_only_reset_bin_always_has_a_valid_transition(self):
        env = TrackingSim.__new__(TrackingSim)
        m = self.cache
        env.motion, env.G_max, env.fail_reset, env.focus_static, env.focus_q = m, 1., 0., False, None
        env.bin_G, env.bin_w_avg = torch.zeros(m.num_bins), torch.ones(m.num_bins)/m.num_bins
        env.device = 'cpu'
        for _ in range(3):
            f, _ = env.sample_frames(100)
            self.assertTrue((f < m.clip_end[m.clip_id[f]]).all())

    def test_shell_entry_uses_cpu_cache_and_honors_overrides(self):
        repo = Path(__file__).parents[1]
        shim = self.root/'python-shim'
        shim.write_text('#!'+sys.executable+'\nimport ast,json,sys\nfrom pathlib import Path\n'
                        'tree=ast.parse(Path("src/blinkmimic/train.py").read_text())\n'
                        'cut=next(n.lineno for n in tree.body if isinstance(n,ast.Import) and any(a.name=="torch" for a in n.names))\n'
                        'sys.argv=["train.py"]+sys.argv[sys.argv.index("blinkmimic.train")+1:]\n'
                        'ns={"__file__":str(Path("src/blinkmimic/train.py").resolve())}\n'
                        'exec("\\n".join(Path("src/blinkmimic/train.py").read_text().splitlines()[:cut-1]),ns)\n'
                        'print(json.dumps(vars(ns["args"])))\n')
        shim.chmod(0o755)
        cmd = ['bash', str(repo/'scripts/train_bones_cpu.sh'), str(self.root/'logs'), '--motion_dir', str(self.root),
               '--motion_cache_gib', '4']
        result = subprocess.run(cmd, env={**os.environ, 'PYTHON': str(shim)}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        for k, v in dict(motion_cache_gib=4, num_envs=4096, buffer_steps=500, pi_batch=2048, wm_batch=2048,
                         eprobe_envs=512, eprobe_share=.15, eprobe_async=0, eprobe_start=1000,
                         eprobe_every=250, eprobe_decay=.5, reset_masked=False).items():
            self.assertEqual(args[k], v)
        invalid = subprocess.run(cmd+['--eprobe_async', '20'], env={**os.environ, 'PYTHON': str(shim)}, capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a CUDA GPU')
    def test_cuda_graph_reads_replaced_pages_through_the_same_device_storage(self):
        import copy
        kin = copy.copy(self.kin)
        for name in ('axes', 'off_p', 'off_R', 'q_lo', 'q_hi'):
            setattr(kin, name, getattr(kin, name).cuda())
        c = CachedMotionLib(self.files, kin, 'cuda:0', load_lafan1_csv,
                            cache_gib=12*self.page_bytes/2**30, page_size=4)
        f = torch.tensor([2, 54], device='cuda:0')
        c.prepare(f, ahead=3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                c.goal(f, 3)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = c.goal(f, 3)
        graph.replay()
        torch.testing.assert_close(out.cpu(), self.dense.goal(f.cpu(), 3), rtol=1e-5, atol=1e-5)
        c.prepare(torch.arange(32, 72, device='cuda:0'))
        f.copy_(torch.tensor([9, 77], device='cuda:0'))
        c.prepare(f, ahead=3)
        graph.replay()
        torch.testing.assert_close(out.cpu(), self.dense.goal(f.cpu(), 3), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
