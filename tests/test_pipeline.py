"""Fast unit tests (no training): data conversion + centre-line geometry. Run: pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lane_extraction import estimate_centerline  # noqa: E402


def test_centerline_recovers_known_curve():
    # left x = 400 + 0.2*(y-300) ... right mirrored; centre should be exactly between them
    ys = np.linspace(320, 600, 8)
    left = np.stack([600 - 0.9 * (ys - 320), ys], 1)
    right = np.stack([700 + 0.9 * (ys - 320), ys], 1)
    xyxy, cls = [], []
    for (x, y), c in [(p, 0) for p in left] + [(p, 1) for p in right]:
        xyxy.append([x - 5, y - 20, x + 5, y]); cls.append(c)
    res = estimate_centerline(np.array(xyxy), np.array(cls), (1280, 720))
    err = np.abs(res.center_curve[:, 0] - 650).max()
    assert err < 1.0, err


def test_metric_mode_depth():
    # a cone 10 m away, fx=900 -> box height = 900*0.325/10 = 29.25 px; two per side
    def box(X, Z):
        h = 900 * 0.325 / Z
        u = 640 + 900 * X / Z
        v = 300 + 900 * 0.9 / Z
        return [u - h / 4, v - h, u + h / 4, v]
    xyxy = np.array([box(-1.75, 8), box(-1.75, 16), box(1.75, 8), box(1.75, 16)])
    res = estimate_centerline(xyxy, np.array([0, 0, 1, 1]), (1280, 720), fx=900)
    assert res.mode == "bev" and np.abs(res.center_curve[:, 0]).max() < 0.05


def test_data_prep_roundtrip(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "tools/make_synthetic_fsoco.py"), "--out", str(src),
                    "--images", "40", "--teams", "10"], check=True)
    subprocess.run([sys.executable, str(ROOT / "data_prep.py"), "--src", str(src), "--out", str(out),
                    "--workers", "1", "--preview", "0"], check=True)
    stats = json.loads((out / "stats.json").read_text())
    assert sum(stats["images"].values()) == stats["status"]["ok"] > 30
    assert (out / "data.yaml").exists()
    lbl = next((out / "labels" / "train").glob("*.txt")).read_text().split()
    assert all(0 <= float(v) <= 1 for v in lbl[1:5])
