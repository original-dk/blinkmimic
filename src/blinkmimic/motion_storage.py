"""Memory/disk preflight and bounded-residency storage for FP32 motion states."""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import shutil

import torch


GIB = 2**30


def memory_budget(meminfo=Path('/proc/meminfo'), cgroup=Path('/sys/fs/cgroup'),
                  membership=Path('/proc/self/cgroup')):
    """Available bytes and effective capacity, including finite ancestor cgroup limits."""
    info = {row.split(':')[0]: int(row.split()[1])*1024 for row in meminfo.read_text().splitlines()}
    available, total = info['MemAvailable'], info['MemTotal']
    roots = {cgroup}
    if membership.exists():
        for row in membership.read_text().splitlines():
            hierarchy, controllers, relative = row.split(':', 2)
            if hierarchy == '0' or 'memory' in controllers.split(','):
                node = cgroup / relative.lstrip('/')
                while node != cgroup and cgroup in node.parents:
                    roots.add(node)
                    node = node.parent
    for root in roots:
        for limit_name, current_name in (('memory.max', 'memory.current'),
                                         ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            limit, current = root/limit_name, root/current_name
            if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
                cap, used = int(limit.read_text()), int(current.read_text())
                total, available = min(total, cap), min(available, max(cap-used, 0))
    return available, total


def check_memory(frames, width, mapped):
    available, total = memory_budget()
    # Global metadata/contact calibration plus a bounded resampling/FK workspace; leave room for the OS.
    reserve = min(4*GIB, total//8)
    scratch = frames*40 + min(frames, 65536)*width*16 + 128*2**20
    required = reserve + scratch + (0 if mapped else frames*width*4)
    if available < required:
        raise MemoryError(f"Motion startup needs {required/GIB:.2f} GiB available including reserve, "
                          f"but RAM/cgroup allows {available/GIB:.2f} GiB. "
                          "Use --motion_cache_dir on a local SSD and free system memory.")
    return available


def check_disk(directory, size, mountinfo=Path('/proc/self/mountinfo'), reusable=0):
    directory = Path(directory).resolve()
    if mountinfo.exists():
        mounts = []
        for row in mountinfo.read_text().splitlines():
            left, right = row.split(' - ', 1)
            mount = left.split()[4].replace('\\040', ' ')
            if str(directory) == mount or str(directory).startswith(mount.rstrip('/')+'/'):
                mounts.append((len(mount), right.split()[0]))
        if mounts and max(mounts)[1] in ('tmpfs', 'ramfs'):
            raise ValueError("motion_cache_dir is on a RAM filesystem; choose a local SSD path")
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    required = max(size-reusable, 0) + max(256*2**20, size//10)
    if free < required:
        raise OSError(f"Motion cache needs {required/GIB:.2f} GiB free disk including reserve; "
                      f"only {free/GIB:.2f} GiB available at {directory}")


class MappedStates:
    """Shared file mapping; flushing and dropping completed ranges bounds startup resident pages."""

    def __init__(self, path, frames, width, create):
        self.path, self.frames, self.width = Path(path), frames, width
        size = frames*width*4
        if create:
            reusable = self.path.stat().st_blocks*512 if self.path.exists() else 0
            check_disk(self.path.parent, size, reusable=reusable)
            with self.path.open('w+b') as f:
                os.posix_fallocate(f.fileno(), 0, size)
        elif self.path.stat().st_size != size:
            raise ValueError(f"Incomplete motion cache: {self.path}")
        with self.path.open('r+b') as f:
            self.mapping = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_WRITE)
        self.tensor = torch.frombuffer(self.mapping, dtype=torch.float32).view(frames, width)
        self.bytes_since_drop = 0

    def release(self, written=False, force=False, nbytes=0):
        self.bytes_since_drop += nbytes
        if force or self.bytes_since_drop >= 64*2**20:
            if written:
                self.mapping.flush()
            self.mapping.madvise(mmap.MADV_DONTNEED)
            self.bytes_since_drop = 0
