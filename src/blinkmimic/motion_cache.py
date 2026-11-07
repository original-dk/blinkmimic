"""FP32 root/joint states on the host, with a bounded, paged GPU working set.

Global frame IDs and the reset sampler cover the entire library. prepare() runs outside CUDA graphs;
captured computation only indexes fixed device tensors through a page table. Replay keeps global IDs,
so evicting a reference never invalidates a recorded transition: it can be loaded again before use.
"""

from __future__ import annotations

import os
import fcntl
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import torch

from .motion import BIN_FRAMES, CONTROL_DT, MotionLib, _gradient, load_npz
from .motion_storage import GIB, MappedStates, check_memory
from .rotations import quat_slerp, quat_to_mat, so3_log


class CachedField:
    """Tensor-like frame indexing; only requested pages reside on the computation device."""

    def __init__(self, owner, name, shape, offset, host=True):
        self.owner, self.name = owner, name
        self.shape = (owner.num_frames, *shape)
        self.device, self.dtype = owner.device, torch.float32
        width = int(np.prod(shape))
        self.host = owner.host[:, offset:offset+width].view(owner.num_frames, *shape) if host else None
        self.data = owner.data[:, offset:offset+width].view(-1, *shape)

    def __getitem__(self, frames):
        slots = self.owner.page_table[frames // self.owner.page_size]
        rows = slots * self.owner.page_size + frames % self.owner.page_size
        return self.data[rows]

    def cpu(self):
        if self.host is None:
            raise RuntimeError(f"{self.name} is cached on the GPU; query prepared frame IDs")
        return self.host


class CachedMotionLib(MotionLib):
    cached = True

    @torch.no_grad()
    def __init__(self, files, kin, device, load_motion, quat_order="wxyz", cache_gib=6, page_size=64, cache_dir=None):
        self.device, self.page_size = torch.device(device), page_size
        self.kin, self.storage = kin, None
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        read = lambda f: load_npz(f, quat_order) if f.endswith(".npz") else load_motion(f)
        manifest, saved = None, None
        if cache_dir is not None:
            key = hashlib.sha256(f"fp32-root-v1:{quat_order}".encode())
            for path in files:
                stat = os.stat(path)
                key.update(f"{os.path.realpath(path)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            folder = Path(cache_dir).resolve()/key.hexdigest()[:24]
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder/'complete.json'
            self._lock = (folder/'lock').open('a+b')
            weakref.finalize(self, self._lock.close)
            ready = manifest.exists()
            try:
                fcntl.flock(self._lock, (fcntl.LOCK_SH if ready else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Motion cache is being built by another process: {folder}") from None
            if ready:
                saved = json.loads(manifest.read_text())
        if saved is not None:
            lengths, self.num_j = saved['lengths'], saved['num_j']
        else:
            lengths = []
            for i, path in enumerate(files):
                p, quat, j, fps = read(path)
                if len(p) < 2 or not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Motion needs two frames and a positive finite fps: {path}")
                lengths.append(int(np.floor((len(p)-1) / fps / CONTROL_DT)) + 1)
                if (i+1) % 10000 == 0:
                    print(f"motion scan: {i+1}/{len(files)} clips", flush=True)
            self.num_j = j.shape[-1] if lengths else 0
        if not lengths or min(lengths) < 2:
            raise ValueError("Motion library needs at least two resampled frames per clip")
        self.num_frames = sum(lengths)
        raw_width = 18 + 2*self.num_j
        available = check_memory(self.num_frames, raw_width, mapped=manifest is not None)
        if manifest is not None:
            self.storage = MappedStates(folder/'states.fp32', self.num_frames, raw_width, create=saved is None)
            self.host = self.storage.tensor
        else:
            self.host = torch.empty(self.num_frames, raw_width, dtype=torch.float32)
        print(f"motion host: FP32 root/joints {self.num_frames*raw_width*4/GIB:.2f} GiB "
              f"({'SSD mapping' if self.storage else 'RAM'}), available RAM/cgroup {available/GIB:.2f} GiB", flush=True)
        self.clip_len = torch.tensor(lengths, device=device)
        starts = np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64)[:-1])))
        self.clip_start = torch.tensor(starts, device=device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).to(device)
        bins_per_clip = (np.array(lengths) + BIN_FRAMES - 1) // BIN_FRAMES
        bin0 = np.concatenate(([0], np.cumsum(bins_per_clip[:-1])))
        bin_clip = np.repeat(np.arange(len(lengths)), bins_per_clip)
        bin_start = starts[bin_clip] + (np.arange(sum(bins_per_clip)) - bin0[bin_clip]) * BIN_FRAMES
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_clip)
        frame_bin = torch.empty(self.num_frames, dtype=torch.long)
        for c, (s, n) in enumerate(zip(starts, lengths)):
            frame_bin[s:s+n] = int(bin0[c]) + torch.arange(n) // BIN_FRAMES
        self.frame_bin = frame_bin.to(device)

        L, J = len(kin.tracked_idx), self.num_j
        shapes = dict(root_p=(3,), root_R=(3, 3), root_v=(3,), root_w=(3,), q=(J,), qd=(J,),
                      body_p=(L, 3), body_R=(L, 3, 3), body_v=(L, 3), body_w=(L, 3),
                      root_v_local=(3,), root_w_local=(3,))
        # Derived body references live only in the GPU cache, never in a full-library CPU allocation.
        width = sum(int(np.prod(s)) for s in shapes.values())
        self.num_pages = (self.num_frames + page_size - 1) // page_size
        self.capacity = min(self.num_pages, int(cache_gib * 2**30) // (page_size * width * 4))
        if self.capacity < 1:
            raise ValueError("motion_cache_gib must fit at least one reference page")
        self.data = torch.empty(self.capacity * page_size, width, device=device)
        # An absent page maps beyond the allocation: missed prefetches fail instead of reading stale data.
        self.page_table = torch.full((self.num_pages,), self.capacity, dtype=torch.long, device=device)
        self.slots = np.full(self.num_pages, -1, dtype=np.int64)
        self.pages = np.full(self.capacity, -1, dtype=np.int64)
        self.stamps = np.zeros(self.capacity, dtype=np.int64)
        self.clock, self.loaded_pages, self.requested_pages = 0, 0, 0
        offset = 0
        for name, shape in shapes.items():
            setattr(self, name, CachedField(self, name, shape, offset, host=offset < raw_width))
            offset += int(np.prod(shape))
        # root_gz aliases the root rotation exactly, as in the resident library.
        self.root_gz = CachedField(self, "root_gz", (), 3+8)
        for path, start, n in ([] if saved is not None else zip(files, starts, lengths)):
            p, quat, j, fps = read(path)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * fps
            i0 = t.floor().long().clamp(max=len(p)-2)
            a = (t-i0.float()).clamp(0, 1)
            p = p[i0] * (1-a[:, None]) + p[i0+1] * a[:, None]
            j = j[i0] * (1-a[:, None]) + j[i0+1] * a[:, None]
            R = quat_to_mat(quat_slerp(quat[i0], quat[i0+1], a))
            v, qd = _gradient(p, CONTROL_DT), _gradient(j, CONTROL_DT)
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2*CONTROL_DT)
            w[0], w[-1] = so3_log(R[1] @ R[0].T) / CONTROL_DT, so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            for name, value in zip(("root_p", "root_R", "root_v", "root_w", "q", "qd"), (p, R, v, w, j, qd)):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite motion data: {path}")
                getattr(self, name).host[start:start+n].copy_(value)
            if self.storage is not None:
                self.storage.release(written=True, nbytes=n*raw_width*4)
            if (self.clip_start.numel() > 10000) and int(start+n) // 1_000_000 != int(start) // 1_000_000:
                print(f"motion preprocess: {start+n:,}/{self.num_frames:,} frames", flush=True)
        if self.storage is not None:
            self.storage.release(written=saved is None, force=True)
            if saved is None:
                temporary = manifest.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(lengths=lengths, num_j=self.num_j)))
                os.replace(temporary, manifest)
                fcntl.flock(self._lock, fcntl.LOCK_SH)
        # Pin only a small transfer buffer. Full body data are computed once per missing cache page.
        batch = min(self.capacity, 256) * page_size
        self.staging = torch.empty(batch, raw_width, pin_memory=self.device.type == "cuda")
        self.transfer = torch.empty(batch, raw_width, device=device)

    def host_state(self, begin, end):
        return tuple(getattr(self, k).host[begin:end].to(self.device)
                     for k in ("root_p", "root_R", "root_v", "root_w", "q", "qd"))

    @property
    def host_bytes(self):
        contact = getattr(self, "ref_contact", None)
        return self.host.numel()*4 + (contact.host.numel()*4 if contact is not None else 0)

    @property
    def device_bytes(self):
        arrays = (self.data, self.page_table, self.clip_len, self.clip_start, self.clip_end, self.clip_id,
                  self.bin_clip, self.bin_start, self.frame_bin, self.transfer)
        return sum(t.numel()*t.element_size() for t in arrays) + (self.ref_contact.data.numel()*4 if hasattr(self, "ref_contact") else 0)

    @torch.no_grad()
    def prepare(self, frames, ahead: int = 0, extra=None):
        """Stage raw FP32 state and compute derived fields only for missing pages, outside graphs."""
        f = frames.detach().reshape(-1).cpu().numpy()
        if ahead < 0 or (len(f) and (f.min() < 0 or f.max() >= self.num_frames)):
            raise ValueError("Invalid motion prefetch range")
        start, end = f // self.page_size, np.minimum(f+ahead, self.num_frames-1) // self.page_size
        span = np.arange((ahead+self.page_size-1)//self.page_size + 1)
        pages = start[:, None] + span
        needed = pages[pages <= end[:, None]]
        if extra is not None:
            other = extra.detach().reshape(-1).cpu().numpy()
            if len(other) and (other.min() < 0 or other.max() >= self.num_frames):
                raise ValueError("Invalid reset reference frame")
            needed = np.concatenate((needed, other // self.page_size))
        needed = np.unique(needed)
        if len(needed) > self.capacity:
            raise RuntimeError(f"Motion batch needs {len(needed)} pages, cache holds {self.capacity}; "
                               "increase --motion_cache_gib or reduce batch/env count")
        self.clock += 1
        self.requested_pages += len(needed)
        missing = needed[self.slots[needed] < 0]
        if not len(missing):
            self.stamps[self.slots[needed]] = self.clock
            return
        available = np.flatnonzero(~np.isin(self.pages, needed))
        victims = available[np.argsort(self.stamps[available], kind="stable")[:len(missing)]]
        old = self.pages[victims]
        old = old[old >= 0]
        self.slots[old] = -1
        self.page_table[torch.from_numpy(old).to(self.device)] = self.capacity
        self.slots[missing], self.pages[victims] = victims, missing
        block = self.staging.shape[0] // self.page_size
        for s in range(0, len(missing), block):
            new, slot = missing[s:s+block], victims[s:s+block]
            src = (new[:, None]*self.page_size + np.arange(self.page_size)).reshape(-1)
            dst = (slot[:, None]*self.page_size + np.arange(self.page_size)).reshape(-1)
            n = len(src)
            torch.index_select(self.host, 0, torch.from_numpy(src.clip(max=self.num_frames-1)), out=self.staging[:n])
            self.transfer[:n].copy_(self.staging[:n], non_blocking=True)
            rows = torch.from_numpy(dst).to(self.device)
            self.data[:, :self.host.shape[1]].index_copy_(0, rows, self.transfer[:n])
            state = tuple(getattr(self, k).data[rows] for k in ("root_p", "root_R", "root_v", "root_w", "q", "qd"))
            for name, value in zip(("body_p", "body_R", "body_v", "body_w"), self.kin.forward(*state)):
                getattr(self, name).data.index_copy_(0, rows, value)
            Rt = state[1].transpose(-1, -2)
            self.root_v_local.data.index_copy_(0, rows, (Rt @ state[2].unsqueeze(-1)).squeeze(-1))
            self.root_w_local.data.index_copy_(0, rows, (Rt @ state[3].unsqueeze(-1)).squeeze(-1))
            if hasattr(self, "ref_contact"):
                self.ref_contact.data.index_copy_(0, torch.from_numpy(dst).to(self.device),
                                                 self.ref_contact.host[src.clip(max=self.num_frames-1)].to(self.device))
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()  # staging may now be safely overwritten
            if self.storage is not None:
                self.storage.release(nbytes=n*self.host.shape[1]*4)
        self.page_table[torch.from_numpy(missing).to(self.device)] = torch.from_numpy(victims).to(self.device)
        self.stamps[self.slots[needed]] = self.clock
        self.loaded_pages += len(missing)

    @torch.no_grad()
    def set_ref_contact(self, feet, margin=0.015, vmax=0.5, q=0.05):
        # A separate small field shares the same page mapping. Refill resident pages after calibration.
        field = CachedField.__new__(CachedField)
        field.owner, field.name = self, "ref_contact"
        field.device, field.dtype, field.shape = self.device, torch.float32, (self.num_frames, len(feet))
        field.host = torch.empty(self.num_frames, len(feet))
        field.data = torch.empty(self.capacity*self.page_size, len(feet), device=self.device)
        z = torch.empty(self.num_frames, len(feet))
        speed = torch.empty_like(z)
        for s in range(0, self.num_frames, 65536):
            e = min(s+65536, self.num_frames)
            body = self.kin.forward(*self.host_state(s, e))
            z[s:e].copy_(body[0][:, feet, 2].cpu())
            speed[s:e].copy_(body[2][:, feet, :2].norm(dim=-1).cpu())
            if self.storage is not None:
                self.storage.release(nbytes=(e-s)*self.host.shape[1]*4)
        starts, ends = self.clip_start.cpu().tolist(), self.clip_end.cpu().tolist()
        for s, e in zip(starts, ends):
            height, velocity = z[s:e+1], speed[s:e+1]
            field.host[s:e+1] = ((height < height.quantile(q, dim=0)+margin) & (velocity < vmax)).float()
        self.ref_contact = field
        self.slots.fill(-1)
        self.pages.fill(-1)
        self.page_table.fill_(self.capacity)
