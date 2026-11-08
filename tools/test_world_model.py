"""CPU regression checks for actor-history world-model inputs and replay sampling."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.dr import XiLayout
from blinkmimic.mdp import MDP, flatten_state, obs_dim, unflatten_state
from blinkmimic.mirror import Mirror
from blinkmimic.networks import WorldModel
from blinkmimic.robot import get_robot
from blinkmimic.rotations import so3_exp
from blinkmimic.sensing import ObsPipeline


def state(B, J):
    return (torch.randn(B, 3), so3_exp(torch.randn(B, 3)), torch.randn(B, 3),
            torch.randn(B, 3), torch.randn(B, J), torch.randn(B, J))


def train_functions(**context):
    # Import the pure training functions without launching the CUDA simulator/CLI.
    tree = ast.parse((Path(__file__).parents[1] / "src/blinkmimic/train.py").read_text())
    names = {"_wm_rollout", "_imag_step", "_wm_norm_data", "_mirror_windows"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, WorldModel=WorldModel, unflatten_state=unflatten_state,
                 flatten_state=flatten_state, CONTROL_DT=0.02, **context)
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), scope)
    return scope


class WorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.J, self.H = 4, 29, 3
        self.S = state(self.B, self.J)
        self.o = torch.randn(self.B, self.H, obs_dim(self.J))
        self.a = torch.randn(self.B, self.J)
        self.xi = torch.zeros(self.B, self.J + 4)
        self.wm = WorldModel(self.J, self.xi.shape[-1], hidden=(32,), hist=self.H)
        self.motion = SimpleNamespace(root_R=torch.eye(3).repeat(20, 1, 1),
                                      clip_end=torch.tensor([19]), clip_id=torch.zeros(20, dtype=torch.long))
        self.mdp = object.__new__(MDP)
        self.mdp.motion, self.mdp.default_q = self.motion, torch.zeros(self.J)
        self.mdp.obs_noise = torch.zeros(obs_dim(self.J))
        self.mdp.obs_dim = obs_dim(self.J)

    def test_inputs_and_outputs(self):
        for delayed in (False, True):
            for structured in (False, True):
                wm = WorldModel(self.J, self.xi.shape[-1], use_prev_action=delayed,
                                structured=structured, hidden=(32,), hist=self.H)
                x = wm.inputs(self.o, self.a, self.a, self.xi)
                torch.testing.assert_close(x[:, :self.H * obs_dim(self.J)], self.o.flatten(1))
                expected = wm.integrate(self.S, torch.randn(self.B, wm.out_dim))
                target = wm.targets(self.S, expected)
                torch.testing.assert_close(wm.integrate(self.S, target), expected)
                torch.testing.assert_close(wm.step(self.S, self.o, self.a, self.a, self.xi),
                                           wm.integrate(self.S, torch.zeros(self.B, wm.out_dim)))
                self.assertEqual(wm.out_dim, 12 + 2 * self.J)

    def test_no_linear_velocity_in_observation(self):
        changed = list(self.S)
        changed[2] = self.S[2] + 100
        f = torch.zeros(self.B, dtype=torch.long)
        torch.testing.assert_close(self.mdp.obs(self.S, self.a, self.xi, f),
                                   self.mdp.obs(changed, self.a, self.xi, f))

    def test_replay_history(self):
        buf = ReplayBuffer(5, 1, "cpu", self.J, self.xi.shape[-1], obs_dim(self.J))
        for t in range(8):
            tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
            tr["o"].fill_(t)
            tr["ep_step"].fill_(t if t < 6 else t - 6)
            buf.add(tr)
        _, h = buf.starts_at(torch.tensor([0, 3, 4]), torch.zeros(3, dtype=torch.long), self.H)
        torch.testing.assert_close(h["o"][:, :, 0], torch.tensor([[3., 3., 3.], [6., 6., 6.], [6., 6., 7.]]))
        buf.data["cut"].fill_(True)
        w = buf.sample_windows(8, 3, self.H)
        self.assertEqual(w["o_hist"].shape, (8, self.H, obs_dim(self.J)))
        torch.testing.assert_close(w["mask"], torch.tensor([[1., 0., 0.]]).expand(8, -1))
        self.assertNotIn("obs_state", w)
        sensed = ReplayBuffer(5, 1, "cpu", self.J, self.xi.shape[-1], obs_dim(self.J), 7)
        sensed.size = 5
        self.assertEqual(sensed.sample_windows(2, 3, self.H)["obs_state"].shape, (2, 7))

    def test_rollout_and_gradients(self):
        torch.nn.init.normal_(self.wm.net[-1].weight, std=0.01)
        self.o.requires_grad_(), self.a.requires_grad_()
        args = SimpleNamespace(wm_window=3, wm_batch=self.B, wm_mirror=0.5)
        scope = train_functions(args=args, wm=self.wm, mdp=self.mdp, motion=self.motion,
                                normalize_xi=lambda x: x, pipe=SimpleNamespace(stateful=False), mirror=None)
        w = dict(s=flatten_state(self.S)[:, None].expand(-1, 3, -1),
                 s1=flatten_state(self.S)[:, None].expand(-1, 3, -1),
                 a=self.a[:, None].expand(-1, 3, -1), a_prev=self.a[:, None].expand(-1, 3, -1),
                 xi=self.xi[:, None].expand(-1, 3, -1), o_hist=self.o,
                 f=torch.full((self.B,), 19, dtype=torch.long))
        result = scope["_wm_rollout"](w)  # includes clip-end clamping
        self.assertEqual(result[0].shape, (self.B, 3, 3))
        sum(x.square().mean() for x in result).backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in self.wm.parameters()))
        self.assertGreater(self.o.grad.abs().sum().item(), 0)
        self.assertGreater(self.a.grad.abs().sum().item(), 0)
        mirror = Mirror(get_robot("g1"), {"joint_offset": slice(1, 30), "com": slice(30, 33)}, "cpu")
        scope["mirror"] = mirror
        mirrored = scope["_wm_rollout"](scope["_mirror_windows"](w, 2))
        for original, reflected in zip(result, mirrored):
            torch.testing.assert_close(original[2:], reflected[2:])
        robot = get_robot("g1")
        layout = XiLayout(robot, "cpu")
        pipe = ObsPipeline(robot, self.mdp, layout, "cpu", 0.005)
        pipe.stateful = True
        w["obs_state"] = pipe.pack(pipe.init_state(self.S, self.xi, None, None))
        scope["pipe"], scope["robot"] = pipe, robot
        sensed = scope["_wm_rollout"](scope["_mirror_windows"](w, 2))
        for plain, filtered in zip(mirrored, sensed):
            torch.testing.assert_close(plain, filtered)
        compiled = torch.compile(scope["_wm_rollout"], backend="eager", fullgraph=True)
        for eager, captured in zip(mirrored, compiled(scope["_mirror_windows"](w, 2))):
            torch.testing.assert_close(eager, captured)

    def test_mirror_observation(self):
        mirror = Mirror(get_robot("g1"), {"joint_offset": slice(1, 30), "com": slice(30, 33)}, "cpu")
        self.motion.root_R[:] = torch.eye(3)
        f = torch.zeros(self.B, dtype=torch.long)
        o = self.mdp.obs(self.S, self.a, self.xi, f)
        expected = self.mdp.obs(mirror.state(self.S), mirror.joints(self.a), mirror.xi(self.xi), f)
        torch.testing.assert_close(mirror.obs(o), expected)
        torch.testing.assert_close(mirror.obs(mirror.obs(self.o)), self.o)


if __name__ == "__main__":
    unittest.main()
