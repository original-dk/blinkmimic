"""Downloader regression checks without network access or changes to local training data."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


FAKE_CURL = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
url = next(x for x in args if x.startswith("https://"))
out = Path(args[args.index("-o") + 1])
mode = os.environ["DOWNLOAD_TEST_MODE"]
if "/api/" in url:
    if mode == "http_error":
        print("curl: (22) The requested URL returned error: 403", file=sys.stderr)
        sys.exit(22)
    if mode == "invalid_json":
        out.write_text("<html>Proxy error</html>")
    elif mode == "empty_listing":
        out.write_text("[]")
    elif mode == "error_object":
        out.write_text('{"error": "Access denied"}')
    else:
        folder = "g1" if url.endswith("/g1") else "robot_description/g1/meshes"
        name = "clip.csv" if folder == "g1" else "pelvis.STL"
        out.write_text(json.dumps([{"type": "file", "path": folder + "/" + name},
                                   {"type": "directory", "path": folder + "/subdir"}]))
else:
    out.write_text("partial" if mode == "partial_file" else "complete")
    if mode == "partial_file":
        sys.exit(18)
'''


class DownloadDataCheck(unittest.TestCase):
    def run_download(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "tools").mkdir()
        (root / "bin").mkdir()
        shutil.copyfile(Path(__file__).with_name("download_data.sh"), root / "tools/download_data.sh")
        for name, source in (("curl", FAKE_CURL), ("uv", '#!/bin/sh\ntouch "$UV_TEST_MARKER"\n')):
            script = root / "bin" / name
            script.write_text(source)
            script.chmod(0o755)
        motions = root / "data/lafan1_g1"
        motions.mkdir(parents=True)
        (motions / "clip.csv").write_text("existing")
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                   DOWNLOAD_TEST_MODE=mode, UV_TEST_MARKER=str(root / "converted"))
        result = subprocess.run(["bash", "tools/download_data.sh"], cwd=root, env=env, capture_output=True, text=True)
        return root, result

    def test_failed_listing_stops_before_conversion(self):
        for mode in ("http_error", "invalid_json", "empty_listing", "error_object"):
            with self.subTest(mode=mode):
                root, result = self.run_download(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "converted").exists())
                self.assertEqual((root / "data/lafan1_g1/clip.csv").read_text(), "existing")
                self.assertNotIn("Traceback", result.stderr)
                self.assertTrue(result.stderr.strip())

    def test_partial_download_preserves_existing_file(self):
        root, result = self.run_download("partial_file")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((root / "data/lafan1_g1/clip.csv").read_text(), "existing")
        self.assertFalse((root / "converted").exists())

    def test_success_downloads_assets_and_runs_conversion(self):
        root, result = self.run_download("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in ("lafan1_g1/clip.csv", "g1_description/meshes/pelvis.STL",
                     "g1_description/g1_29dof_rev_1_0.urdf"):
            self.assertEqual((root / "data" / path).read_text(), "complete")
        self.assertTrue((root / "converted").exists())
        self.assertFalse((root / "data/lafan1_g1/subdir").exists())


if __name__ == "__main__":
    unittest.main()
