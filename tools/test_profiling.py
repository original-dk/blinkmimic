"""Check timing windows and that disabled / CPU timing never synchronizes CUDA."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blinkmimic.profiling import TrainingTimer


class TrainingTimerCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "phase_times.csv"

    def test_disabled_and_outside_window_do_not_sync_or_write(self):
        for start, steps, iteration in ((0, 0, 0), (10, 2, 9), (10, 2, 12)):
            timer = TrainingTimer(str(self.path), "cuda:0", start, steps)
            with patch("torch.cuda.synchronize") as sync:
                timer.begin(iteration)
                for phase in timer.phases:
                    timer.mark(phase)
                timer.finish()
                timer.close()
                sync.assert_not_called()
            self.assertFalse(self.path.exists())

    def test_cpu_window_phase_totals_and_append(self):
        counts = dict.fromkeys(TrainingTimer.counts, 0)
        counts.update(collect_steps=3, world_model_updates=5, policy_updates=1)
        for iteration in (10, 11):
            timer = TrainingTimer(str(self.path), "cpu", 10, 2)
            with patch("torch.cuda.synchronize") as sync, patch("blinkmimic.profiling.time.perf_counter", side_effect=range(8)):
                timer.begin(iteration)
                for phase in timer.phases:
                    timer.mark(phase)
                timer.finish(**counts)
                timer.close()
                sync.assert_not_called()
        with self.path.open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["iteration"]) for row in rows], [11, 12])
        for row in rows:
            self.assertEqual(sum(float(row[k]) for k in TrainingTimer.phases), float(row["total_ms"]))
            self.assertEqual(float(row["total_ms"]), 7000)
            self.assertEqual(int(row["collect_steps"]), 3)


if __name__ == "__main__":
    unittest.main()
