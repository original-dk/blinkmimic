"""CPU checks for the auxiliary velocity fit, collection alignment, and checkpoint compatibility."""

import ast
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.mdp import flatten_state, obs_dim, unflatten_state
from blinkmimic.mirror import Mirror
from blinkmimic.robot import get_robot
from blinkmimic.networks import Policy, VelocityEstimator, WorldModel
from blinkmimic import rotations


TRAIN = ast.parse((Path(__file__).parents[1] / "src/blinkmimic/train.py").read_text())


def train_functions(**context):
    # Load real training functions without starting the CLI/CUDA simulator.
    names = {"velocity_update", "collect", "_collect_core", "load_opt_states", "_wm_norm_data", "_mix_velocity", "_velocity_input", "velocity_ema_update",
             "_wm_rollout", "_imag_step", "_mirror_windows", "_wm_err", "wm_update"}
    body = [n for n in TRAIN.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, nn=nn, rotations=rotations, WorldModel=WorldModel,
                 unflatten_state=unflatten_state, normalize_xi=lambda x: x, **context)
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), scope)
    return scope


class Sim:
    def __init__(self, N, J):
        self.N, self.J = N, J
        self.f = torch.zeros(N, dtype=torch.long)
        self.ep_step = torch.arange(N)
        self.a_prev = torch.zeros(N, J)
        self.R = rotations.so3_exp(torch.tensor([[0., 0., 1.], [0., 0., -0.5]]))
        self.last_s = None

    def observe(self):
        return self.f[:, None].float().expand(-1, obs_dim(self.J)) + 1

    def step(self, a):
        p = self.f[:, None].float().expand(-1, 3)
        v = torch.stack((self.f + 1, self.f + 2, self.f + 3), -1).float()
        S = (p, self.R, v, torch.zeros_like(v), self.a_prev, torch.zeros_like(a))
        self.last_s = tuple(x.clone() for x in S)
        s = flatten_state(S)
        out = dict(s=s, s1=s.clone(), a=a, a_prev=self.a_prev, xi=torch.zeros(self.N, 1),
                   r=torch.ones(self.N), f=self.f, done=torch.zeros(self.N, dtype=torch.bool),
                   term=torch.zeros(self.N, dtype=torch.bool))
        self.f, self.ep_step, self.a_prev = self.f + 1, self.ep_step + 1, a
        return out

    step_core = step


class VelocityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.N, self.J, self.H = 2, 2, 3
        self.O = obs_dim(self.J)

    def setup_collection(self):
        pi = Policy(self.O, 5, self.J, hist=self.H, enc_hidden=(8,), dec_hidden=(8,))
        wm = WorldModel(self.J, 1, hidden=(8,))
        vel = VelocityEstimator(self.O, self.H)
        env = Sim(self.N, self.J)
        hist = torch.zeros(self.N, self.H, self.O)
        scope = train_functions(args=SimpleNamespace(collect_graph=False, num_envs=self.N, action_noise=0.1,
                                                     compile_more=False, reset_masked=False),
                                pi=pi, wm=wm, vel=vel, vel_avg=copy.deepcopy(vel).requires_grad_(False),
                                vel_mix=torch.tensor(0.), env=env, obs_hist=hist, sim_steps=0,
                                recorded=[],
                                motion=SimpleNamespace(goal=lambda f, k: torch.zeros(len(f), 5)), K=1,
                                CG=dict(obs_hist=hist.clone(), f=env.f.clone(),
                                        ep_step=env.ep_step.clone(), a_prev=env.a_prev.clone()))
        scope["buffer"] = SimpleNamespace(add=scope["recorded"].append, add_device=scope["recorded"].append)
        return scope

    def test_collection_alignment_and_reset(self):
        for graphed_body in (False, True):
            scope = self.setup_collection()
            env = scope["env"]
            seen = []
            scope["pi"].register_forward_pre_hook(lambda m, inp: seen.append(inp[0].clone()))
            for _ in range(2):
                env.ep_step[0] = 0
                if graphed_body:
                    scope["_collect_core"]()
                    h = scope["CG"]["obs_hist"]
                else:
                    stats = {k: [] for k in ("r", "ep_len", "ep_ret", "term", "done")}
                    scope["collect"](1, stats)
                    h = scope["obs_hist"]
                torch.testing.assert_close(h, seen[-1])
                torch.testing.assert_close(h[0], h[0, -1:].expand_as(h[0]))
                recorded = scope["recorded"][-1]
                torch.testing.assert_close(recorded["o"], h[:, -1])
                torch.testing.assert_close(recorded["s"], flatten_state(env.last_s))
                self.assertFalse(recorded["o"].requires_grad)

    def test_fit_and_isolation(self):
        vel = VelocityEstimator(self.O, self.H)
        h = torch.randn(64, self.H, self.O, requires_grad=True)
        target = h.detach()[:, -1, :3] * 0.3
        vel.obs_norm.update(h.detach())
        opt = torch.optim.Adam(vel.parameters(), lr=0.003)
        S = (torch.zeros(64, 3), torch.eye(3).expand(64, -1, -1), target,
             torch.zeros(64, 3), torch.zeros(64, self.J), torch.zeros(64, self.J))
        data = ({"s": flatten_state(S)}, {"o": h.detach()})
        scope = train_functions(vel=vel, opt_vel=opt, Hh=self.H, args=SimpleNamespace(wm_batch=64),
                                vel_rng=torch.Generator().manual_seed(1),
                                buffer=SimpleNamespace(sample_starts=lambda *a, **kw: data),
                                obs_hist=torch.full_like(h, float("nan")))
        original = Policy(self.O, 5, self.J, hist=self.H, enc_hidden=(8,), dec_hidden=(8,))
        before = copy.deepcopy(original.state_dict())
        rng = torch.get_rng_state().clone()
        first = scope["velocity_update"]().item()
        for _ in range(60):
            error = scope["velocity_update"]()
        self.assertLess(error.item(), first * 0.2)
        self.assertIsNone(h.grad)
        torch.testing.assert_close(original.state_dict(), before)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        clone = VelocityEstimator(self.O, self.H)
        clone.load_state_dict(vel.state_dict())
        torch.testing.assert_close(clone(h), vel(h))

    def test_buffer_sampling_and_velocity_source(self):
        buf = ReplayBuffer(5, 1, "cpu", self.J, 1, self.O)
        for t in range(8):
            tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
            R = rotations.so3_exp(torch.tensor([[0., 0., 0.7]]))
            S = (torch.zeros(1, 3), R, torch.tensor([[float(t), 2., 3.]]),
                 torch.zeros(1, 3), torch.zeros(1, self.J), torch.zeros(1, self.J))
            tr["s"] = flatten_state(S)
            tr["o"].fill_(t)
            tr["ep_step"].fill_(t if t < 6 else t - 6)
            buf.add(tr)
        _, h = buf.starts_at(torch.tensor([0, 3, 4]), torch.zeros(3, dtype=torch.long), self.H)
        torch.testing.assert_close(h["o"][:, :, 0], torch.tensor([[3., 3., 3.], [6., 6., 6.], [6., 6., 7.]]))
        self.assertNotIn("obs_state", buf.data)
        for device_ptr in (False, True):
            if device_ptr:
                buf.use_device_ptr(device_size=True)
            rng = torch.get_rng_state().clone()
            s, h = buf.sample_starts(16, self.H, generator=torch.Generator().manual_seed(7))
            torch.testing.assert_close(torch.get_rng_state(), rng)
            S = unflatten_state(s["s"])
            torch.testing.assert_close(S[2][:, 0], h["o"][:, -1, 0])
        vel = VelocityEstimator(self.O, self.H)
        predictions = []
        vel.register_forward_hook(lambda m, inp, out: predictions.append(out.detach().clone()))
        scope = train_functions(vel=vel, opt_vel=torch.optim.Adam(vel.parameters()),
                                args=SimpleNamespace(wm_batch=16), Hh=self.H,
                                vel_rng=torch.Generator().manual_seed(7), buffer=buf)
        error = scope["velocity_update"]()
        target = rotations.mv3(S[1].transpose(-1, -2), S[2])
        torch.testing.assert_close(error, (predictions[-1] - target).square().mean().sqrt())

    def test_window_histories(self):
        buf = ReplayBuffer(5, 1, "cpu", self.J, 1, self.O)
        for t in range(8):
            tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
            tr["o"].fill_(t)
            tr["ep_step"].fill_(t if t < 6 else t - 6)
            buf.add(tr)
        w = buf.sample_windows(16, 3, self.H)
        h = w["o_hist"][..., 0]
        for b in range(16):
            for j in range(3):
                t = int(h[b, j, -1])
                ep_step = t if t < 6 else t - 6
                expected = [t - min(back, ep_step, t - 3) for back in (2, 1, 0)]
                torch.testing.assert_close(h[b, j], torch.tensor(expected, dtype=h.dtype))

    def test_input_mixture_and_integration(self):
        scope = self.setup_collection()
        wm, vel_avg = scope["wm"], scope["vel_avg"]
        S = (torch.randn(2, 3), rotations.so3_exp(torch.randn(2, 3)),
             torch.randn(2, 3, requires_grad=True), torch.randn(2, 3),
             torch.randn(2, self.J), torch.randn(2, self.J))
        a, xi = torch.randn(2, self.J), torch.zeros(2, 1)
        h = torch.randn(2, self.H, self.O, requires_grad=True)
        base = wm.inputs(S, a, a, xi)
        mixed = scope["_velocity_input"](S, h)
        torch.testing.assert_close(wm.inputs(S, a, a, xi, mixed), base)
        scope["vel_mix"].fill_(1.)
        predicted = scope["_velocity_input"](S, h)
        torch.testing.assert_close(predicted, vel_avg(h))
        features = wm.inputs(S, a, a, xi, predicted)
        torch.testing.assert_close(features[..., :7], base[..., :7])
        torch.testing.assert_close(features[..., 10:], base[..., 10:])
        # A zero-output wm integrates the original state even with a different velocity input.
        result = wm.step(S, a, a, xi, predicted)
        torch.testing.assert_close(result, S)
        result[2].sum().backward()
        self.assertIsNone(h.grad)
        self.assertTrue(all(p.grad is None for p in vel_avg.parameters()))
        torch.testing.assert_close(S[2].grad, torch.ones_like(S[2]))

    def test_velocity_ema_and_ramp(self):
        live = VelocityEstimator(self.O, self.H)
        avg = copy.deepcopy(live).requires_grad_(False)
        before = [p.clone() for p in avg.parameters()]
        with torch.no_grad():
            for p in live.parameters():
                p.add_(1.)
            live.obs_norm.mean.fill_(3.)
        scope = train_functions(vel=live, vel_avg=avg, vel_mix=torch.tensor(0.), vel_steps=999)
        scope["velocity_ema_update"]()
        self.assertEqual(scope["vel_mix"].item(), 0.)
        for old, smooth in zip(before, avg.parameters()):
            torch.testing.assert_close(smooth, old + 0.005)
        torch.testing.assert_close(avg.obs_norm.mean, live.obs_norm.mean)
        for _ in range(1000):
            scope["velocity_ema_update"]()
        self.assertAlmostEqual(scope["vel_mix"].item(), 1 - 0.999 ** 1000, places=5)
        self.assertTrue(all(p.grad is None for p in live.parameters()))

    def test_wm_rollout_and_mirror(self):
        scope = self.setup_collection()
        wm = scope["wm"]
        torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
        scope["args"].wm_window = 3
        S = (torch.randn(2, 3), torch.eye(3).expand(2, -1, -1), torch.randn(2, 3),
             torch.randn(2, 3), torch.randn(2, self.J), torch.randn(2, self.J))
        s = flatten_state(S)[:, None].expand(-1, 3, -1)
        a = torch.randn(2, 3, self.J)
        xi = torch.zeros(2, 3, 1)
        v_est = torch.randn(2, 3, 3, requires_grad=True)
        scope["vel_mix"].fill_(0.5)
        result = scope["_wm_rollout"](s, a, a, xi, v_est)
        sum(t.square().mean() for t in result).backward()
        self.assertIsNone(v_est.grad)
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in wm.parameters()))
        compiled = torch.compile(scope["_wm_rollout"], backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(s, a, a, xi, v_est), result)
        mirror = Mirror(get_robot("g1"), {"joint_offset": slice(1, 30), "com": slice(30, 33)}, "cpu")
        scope["mirror"] = mirror
        scope["flatten_state"] = flatten_state
        J = 29
        S = (torch.randn(2, 3), torch.eye(3).expand(2, -1, -1), torch.randn(2, 3),
             torch.randn(2, 3), torch.randn(2, J), torch.randn(2, J))
        w = dict(s=flatten_state(S), s1=flatten_state(S), a=torch.randn(2, J),
                 a_prev=torch.randn(2, J), xi=torch.zeros(2, 33), v_est=v_est[:, 0])
        mirrored = scope["_mirror_windows"](w, 1)
        torch.testing.assert_close(mirrored["v_est"][0], v_est[0, 0] @ mirror.M)
        torch.testing.assert_close(mirrored["v_est"][1], v_est[1, 0])

    def test_wm_update_uses_frozen_estimator(self):
        scope = self.setup_collection()
        scope["args"].wm_batch, scope["args"].wm_window = 4, 3
        scope["args"].wm_mirror = 0.
        scope["Hh"], scope["mirror"] = self.H, None
        wm = scope["wm"]
        scope["opt_wm"] = torch.optim.Adam(wm.parameters(), lr=0.001)
        scope["ERR_FLOOR"] = torch.full((wm.out_dim,), 0.001)
        scope["wm_err"], scope["wm_rollout"] = scope["_wm_err"], scope["_wm_rollout"]
        buf = ReplayBuffer(5, 2, "cpu", self.J, 1, self.O)
        for t in range(5):
            tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
            S = (torch.full((2, 3), float(t)), torch.eye(3).expand(2, -1, -1), torch.ones(2, 3),
                 torch.zeros(2, 3), torch.zeros(2, self.J), torch.zeros(2, self.J))
            S1 = (S[0] + 0.1,) + S[1:]
            tr["s"], tr["s1"] = flatten_state(S), flatten_state(S1)
            tr["ep_step"].fill_(t)
            tr["o"].normal_()
            buf.add(tr)
        scope["buffer"] = buf
        scope["vel_mix"].fill_(0.5)
        before = wm.net[-1].weight.detach().clone()
        loss, last = scope["wm_update"]()
        self.assertTrue(loss.isfinite() and last.isfinite())
        self.assertFalse(torch.equal(before, wm.net[-1].weight))
        self.assertTrue(all(p.grad is None for p in scope["vel_avg"].parameters()))
        self.assertTrue(all(p.grad is None for p in scope["vel"].parameters()))

    def test_imagination_uses_same_mixture(self):
        scope = self.setup_collection()
        torch.nn.init.normal_(scope["wm"].net[-1].weight, std=0.01)
        scope["mdp"] = SimpleNamespace(obs=lambda S, *a: S[0][:, :1].expand(-1, self.O),
                                        add_noise=lambda x: x)
        S = (torch.randn(2, 3), torch.eye(3).expand(2, -1, -1), torch.randn(2, 3),
             torch.randn(2, 3), torch.randn(2, self.J), torch.randn(2, self.J))
        h = torch.randn(2, self.H, self.O)
        a, xi, f, g = torch.zeros(2, self.J), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long), torch.zeros(2, 5)
        scope["vel_mix"].fill_(1.)
        result = scope["_imag_step"](S, h, a, f, xi, xi, g)
        expected_a = scope["pi"](h, g)
        expected = scope["wm"].step(S, expected_a, a, xi, scope["vel_avg"](h))
        torch.testing.assert_close(result[0], expected)
        sum(x.square().mean() for x in result[0]).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in scope["pi"].parameters()))
        self.assertTrue(all(p.grad is None for p in scope["vel_avg"].parameters()))

    def test_resume_ema_state(self):
        resume = next(n for n in TRAIN.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.resume")
        names = {"vel_avg.load_state_dict", "vel_mix.copy_"}
        nodes = [n for n in resume.body if
                 (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) in names)
                 or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "vel_steps" for t in n.targets))]
        code = compile(ast.Module(body=nodes, type_ignores=[]), "train.py", "exec")
        vel = VelocityEstimator(self.O, self.H)
        avg = copy.deepcopy(vel).requires_grad_(False)
        with torch.no_grad():
            for p in avg.parameters():
                p.add_(0.2)
        scope = dict(vel=vel, vel_avg=copy.deepcopy(vel), vel_mix=torch.tensor(0.), vel_steps=0,
                     ck=dict(vel_avg=avg.state_dict(), vel_mix=torch.tensor(0.6), vel_steps=2500))
        exec(code, scope)
        torch.testing.assert_close(scope["vel_avg"].state_dict(), avg.state_dict())
        self.assertAlmostEqual(scope["vel_mix"].item(), 0.6, places=6)
        self.assertEqual(scope["vel_steps"], 2500)
        scope.update(ck={}, vel_mix=torch.tensor(0.))
        exec(code, scope)
        torch.testing.assert_close(scope["vel_avg"].state_dict(), vel.state_dict())
        self.assertEqual(scope["vel_mix"].item(), 0.)
        self.assertEqual(scope["vel_steps"], 0)

    def test_initialization_preserves_rng(self):
        node = next(n for n in TRAIN.body if isinstance(n, ast.With)
                    and "fork_rng" in ast.unparse(n.items[0].context_expr))
        node = copy.deepcopy(node)
        # The training executable uses cuda:0; adapt only its device list for this CPU check.
        node.items[0].context_expr.keywords[0].value = ast.List(elts=[], ctx=ast.Load())
        ast.fix_missing_locations(node)
        scope = dict(torch=torch, VelocityEstimator=VelocityEstimator, dev="cpu", Hh=self.H,
                     mdp=SimpleNamespace(obs_dim=self.O))
        before = torch.get_rng_state().clone()
        exec(compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"), scope)
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_optimizer_resume(self):
        opts = {name: torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=torch.tensor(0.002))
                for name in ("opt_wm", "opt_pi", "opt_v", "opt_vel")}
        for opt in opts.values():
            p = opt.param_groups[0]["params"][0]
            p.square().backward()
            opt.step()
        ck = {k: copy.deepcopy(o.state_dict()) for k, o in opts.items()}
        scope = train_functions(args=SimpleNamespace(vel_lr=0.007), **opts)
        scope["load_opt_states"](ck)
        self.assertEqual(opts["opt_vel"].param_groups[0]["lr"], 0.007)
        self.assertEqual(next(iter(opts["opt_vel"].state.values()))["step"].item(), 1)
        del ck["opt_vel"]  # original main checkpoints remain loadable
        scope["load_opt_states"](ck)
        self.assertEqual(opts["opt_vel"].param_groups[0]["lr"], 0.007)

    def test_velocity_lr_independent_of_schedule(self):
        loop = next(n for n in TRAIN.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and n.target.id == "it")
        schedule = next(n for n in loop.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                        and n.target.id == "o")
        opts = {name: SimpleNamespace(param_groups=[{"lr": 0.007}])
                for name in ("opt_wm", "opt_pi", "opt_v", "opt_vel")}
        scope = dict(torch=torch, lr=0.00001, pi_mult=4.0, **opts)
        exec(compile(ast.Module(body=[schedule], type_ignores=[]), "train.py", "exec"), scope)
        self.assertEqual(opts["opt_vel"].param_groups[0]["lr"], 0.007)
        self.assertEqual(opts["opt_wm"].param_groups[0]["lr"], 0.00001)
        self.assertEqual(opts["opt_pi"].param_groups[0]["lr"], 0.00004)

    def test_logging_and_checkpoints(self):
        # Execute the actual log/checkpoint expressions without simulator or file I/O.
        nodes = list(ast.walk(TRAIN))
        header = next(n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "write" and n.args and isinstance(n.args[0], ast.Constant)
                      and str(n.args[0].value).startswith("iteration,wall_h"))
        self.assertEqual(header.args[0].value.strip().split(",")[-1], "vel_rmse")
        scope = dict(LOG_EXTRA=True, LOG_VEL=True, lr=0.001, pi_mult=1.0, NONFINITE=0,
                     agg={"vel": [torch.tensor(0.2), torch.tensor(0.4)]},
                     mean=lambda x: sum(float(v) for v in x) / len(x), row=list(range(16)), log_f=io.StringIO())
        for n in nodes:
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                target = n.targets[0] if isinstance(n, ast.Assign) else n.target
                if isinstance(target, ast.Name) and target.id in {"extra", "vel_rmse"}:
                    exec(compile(ast.Module(body=[n], type_ignores=[]), "train.py", "exec"), scope)
        self.assertAlmostEqual(scope["extra"][-1], 0.3, places=6)
        self.assertEqual(len(scope["row"] + scope["extra"]), len(header.args[0].value.strip().split(",")))
        # All three checkpoint paths must persist the estimator; full checkpoints also persist Adam.
        for n in nodes:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"ck", "small"} for t in n.targets):
                if isinstance(n.value, ast.Dict):
                    keys = [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
                    if "pi" in keys:
                        self.assertIn("vel", keys)
                        self.assertTrue({"vel_avg", "vel_mix", "vel_steps"}.issubset(keys))
                        if "opt_wm" in keys:
                            self.assertIn("opt_vel", keys)

    def test_compile_and_shared_frame_normalization(self):
        vel = VelocityEstimator(self.O, self.H)
        vel.obs_norm.update(torch.randn(8, self.O))
        h = torch.randn(2, self.H, self.O)
        compiled = torch.compile(vel, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(h), vel(h))
        self.assertEqual(vel.obs_norm.mean.numel(), self.O)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
