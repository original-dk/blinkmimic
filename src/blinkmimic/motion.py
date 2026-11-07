"""GPU motion library: FP32 root positions, packed FP16 states, and reference FK only on requested frames."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import mat_to_6d, mm3, mv3, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


GOAL_Q_OFFSET = 15  # index of the joint positions q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)


def load_npz(path: str, quat_order: str = "wxyz") -> MotionArrays:
    """Packed qpos (root xyz, root quaternion, joints) and scalar fps; return wxyz internally."""
    with np.load(path) as data:
        raw, fps = data["qpos"], float(data["fps"].item())
    quat_columns = {"wxyz": [3, 4, 5, 6], "xyzw": [6, 3, 4, 5]}[quat_order]
    return raw[:, :3], raw[:, quat_columns], raw[:, 7:], fps


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str, load_motion: Callable[[str], MotionArrays],
                 quat_order: str = "wxyz", storage_dtype: torch.dtype = torch.float16):
        self.device, self.kin = device, kin
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        read = lambda f: load_npz(f, quat_order) if f.endswith(".npz") else load_motion(f)
        # Two bounded passes: discover lengths, then write one clip at a time directly into final storage.
        lens = []
        for f in files:
            p, quat, j, src_fps = read(f)
            if len(p) < 2 or not np.isfinite(src_fps) or src_fps <= 0:
                raise ValueError(f"Motion needs at least two frames and a positive finite fps: {f}")
            lens.append(int(np.floor((len(p) - 1) / src_fps / CONTROL_DT)) + 1)
        if not lens or min(lens) < 2:
            raise ValueError("Motion library needs at least two resampled frames per clip")
        self.num_j, self.num_frames = j.shape[-1], sum(lens)
        lengths = np.array(lens, dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(lengths[:-1])))
        self.clip_len = torch.tensor(lengths, device=device)
        self.clip_start = torch.tensor(starts, device=device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.root_p = torch.empty(self.num_frames, 3, dtype=torch.float32, device=device)
        self.rest = torch.empty(self.num_frames, 10 + 2 * self.num_j, dtype=storage_dtype, device=device)
        for f, start, n in zip(files, starts, lens):
            p, quat, j, src_fps = read(f)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            # resample to 50 Hz
            T_src = p.shape[0]
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            self.root_p[start:start+n].copy_(p)
            packed = torch.cat((quat, v, w, j, qd), -1).to(storage_dtype)
            if not torch.isfinite(p).all() or not torch.isfinite(packed).all():
                raise ValueError(f"Non-finite motion data or overflow in {storage_dtype}: {f}")
            self.rest[start:start+n].copy_(packed)
        # Views share one allocation; rotations and local velocities are decoded only for requested frames.
        self.root_quat, self.root_v, self.root_w, self.q, self.qd = self.rest.split((4, 3, 3, self.num_j, self.num_j), -1)
        nb = (lengths + BIN_FRAMES - 1) // BIN_FRAMES
        bin_clip = np.repeat(np.arange(len(lens)), nb)
        bin_k = np.concatenate([np.arange(n) for n in nb])
        bin_start = starts[bin_clip] + bin_k * BIN_FRAMES
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.clip_bin0 = torch.tensor(np.concatenate(([0], np.cumsum(nb[:-1]))), device=device)
        self.num_bins = int(nb.sum())

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kin: Kinematics, device: str, quat_order: str = "wxyz") -> MotionLib:
        files = sorted({f for pattern in (robot.motion_glob, "*.npz")
                        for f in glob.glob(os.path.join(path, "**", pattern), recursive=True)})
        if not files:
            raise FileNotFoundError(f"No motions matching {robot.motion_glob} or *.npz in {path}")
        motion = MotionLib(files, kin, device, robot.load_motion, quat_order)
        motion.names = [os.path.splitext(os.path.relpath(f, path))[0] for f in files]
        return motion

    @torch.no_grad()
    def set_ref_contact(self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, q: float = 0.05) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        # Only foot scalars live on the CPU during preprocessing; no full-library body tensors on GPU.
        z = torch.empty(self.num_frames, len(feet))
        vh, contact = torch.empty_like(z), torch.empty_like(z, dtype=torch.bool)
        for s in range(0, self.num_frames, 65536):
            e = min(s + 65536, self.num_frames)
            bp, _, bv, _ = self.body_state(torch.arange(s, e, device=self.device))
            z[s:e], vh[s:e] = bp[:, feet, 2].cpu(), bv[:, feet, :2].norm(dim=-1).cpu()
        for s, n in zip(self.clip_start.tolist(), self.clip_len.tolist()):
            floor = z[s:s+n].quantile(q, dim=0)
            contact[s:s+n] = (z[s:s+n] < floor + margin) & (vh[s:s+n] < vmax)
        self.ref_contact = contact.to(self.device)

    def clip_of(self, f: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.clip_start, f.contiguous(), right=True) - 1

    def bin_of(self, f: torch.Tensor) -> torch.Tensor:
        c = self.clip_of(f)
        return self.clip_bin0[c] + torch.div(f - self.clip_start[c], BIN_FRAMES, rounding_mode="floor")

    @property
    def storage_bytes(self) -> int:
        tensors = (self.root_p, self.rest, self.clip_start, self.clip_end, self.clip_len,
                   self.clip_bin0, self.bin_clip, self.bin_start)
        return sum(x.numel() * x.element_size() for x in tensors) + (
            self.ref_contact.numel() * self.ref_contact.element_size() if hasattr(self, "ref_contact") else 0)

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.q.shape[-1])

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_of(f)]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        p, R, _, _, _, _ = self.ref_state(f)
        fp, fR, fv, fw, fq, fqd = self.ref_state(fut)
        H = yaw_mat(R)  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = mv3(Ht, fp - p.unsqueeze(1))
        rot = mat_to_6d(mm3(Ht, fR))
        Rt = fR.transpose(-1, -2)
        feats = torch.cat(
            (dp, rot, mv3(Rt, fv), mv3(Rt, fw), fq, fqd * 0.1), -1
        )  # (B,K,15+2J)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        quat, v, w, q, qd = self.rest[f].float().split((4, 3, 3, self.num_j, self.num_j), -1)
        R = quat_to_mat(F.normalize(quat, dim=-1))
        return self.root_p[f], R, v, w, q, qd

    def body_state(self, f: torch.Tensor):
        return self.kin.forward(*self.ref_state(f))

    def reference_arrays(self) -> dict:
        """CPU reference arrays for recording; never materialize a full-library rotation tensor on GPU."""
        R = quat_to_mat(F.normalize(self.root_quat.cpu().float(), dim=-1))
        return dict(ref_root_p=self.root_p.cpu().numpy(), ref_root_R=R.numpy(), ref_q=self.q.cpu().float().numpy())


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
