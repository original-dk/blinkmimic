"""Opt-in synchronized wall times for training phases (host work and device completion)."""

from __future__ import annotations

import csv
import os
import time

import torch


class TrainingTimer:
    phases = ("schedule_ms", "probe_ms", "collect_ms", "world_model_ms", "policy_critic_ms", "logging_ms", "checkpoint_ms")
    counts = ("collect_steps", "world_model_updates", "policy_updates", "probe_sweeps", "checkpoint_submissions")

    def __init__(self, path: str, device: str, start: int, steps: int):
        self.path, self.device, self.start, self.stop = path, torch.device(device), start, start + steps
        self.active = False
        self.file = self.writer = None
        self.totals = dict.fromkeys(self.phases, 0.0)
        self.rows = 0

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def begin(self, iteration: int):
        self.active = self.start <= iteration < self.stop
        if not self.active:
            return
        if self.file is None:
            exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            self.file = open(self.path, "a", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=("iteration", *self.counts, *self.phases, "total_ms"))
            if not exists:
                self.writer.writeheader()
        self.row = {"iteration": iteration + 1}
        self.t0 = self.last = self._clock()

    def mark(self, phase: str):
        if self.active:
            now = self._clock()
            self.row[phase] = 1000 * (now - self.last)
            self.last = now

    def finish(self, **counts):
        if not self.active:
            return
        self.row.update(counts)
        self.row["total_ms"] = 1000 * (self.last - self.t0)
        self.writer.writerow(self.row)
        self.file.flush()
        for phase in self.phases:
            self.totals[phase] += self.row[phase]
        self.rows += 1
        self.active = False

    def close(self):
        if self.file is not None:
            self.file.close()
            means = " | ".join(f"{k.removesuffix('_ms')} {v / self.rows:.2f} ms" for k, v in self.totals.items()) if self.rows else "no rows"
            print(f"--profile_training: {self.rows} iterations | {means} | {self.path}", flush=True)
