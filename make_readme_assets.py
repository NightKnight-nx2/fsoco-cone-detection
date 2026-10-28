"""Produce every README asset in one go (after training): metrics, plots, demo GIFs, lane image.

Runs ``evaluate.py``, ``inference.py``, ``tracking.py`` and ``lane_extraction.py`` and writes into ``docs/``:

    docs/metrics.md, docs/confusion_matrix_normalized.png, docs/predictions_grid.jpg,
    docs/demo_detection.gif, docs/demo_tracking.gif, docs/lane_extraction.jpg, docs/class_distribution.png

and replaces the table between ``<!-- METRICS_START -->`` / ``<!-- METRICS_END -->`` in README.md.

Example::

    python make_readme_assets.py --weights runs/fsoco_yolo11n/weights/best.pt \
        --data data/fsoco_yolo/data.yaml --video my_drive.mp4 --lane-image data/fsoco_yolo/images/test/<x>.jpg
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("\n$", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--video", required=True, help="driving video (or frame folder) for the demo GIFs")
    ap.add_argument("--lane-image", default="", help="single frame for the centre-line figure (default: first video frame)")
    ap.add_argument("--fx", type=float, default=None, help="focal length px for the top-down lane view")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    docs = HERE / "docs"
    docs.mkdir(exist_ok=True)
    res = HERE / "results"

    run("evaluate.py", "--weights", a.weights, "--data", a.data, "--split", "test", "--out", str(res), "--device", a.device)
    for f in ("confusion_matrix_normalized.png", "predictions_grid.jpg", "BoxPR_curve.png", "metrics.md", "metrics.json"):
        if (res / f).exists():
            shutil.copy(res / f, docs / f)
    cd = Path(a.data).parent / "class_distribution.png"
    if cd.exists():
        shutil.copy(cd, docs / cd.name)

    run("inference.py", "--weights", a.weights, "--source", a.video, "--out", str(HERE / "outputs/detection.mp4"),
        "--gif", str(docs / "demo_detection.gif"), "--device", a.device)
    run("tracking.py", "--weights", a.weights, "--source", a.video, "--out", str(HERE / "outputs/tracking.mp4"),
        "--gif", str(docs / "demo_tracking.gif"), "--device", a.device)

    lane_img = a.lane_image
    if not lane_img:
        import cv2

        from common import imwrite, iter_source

        frame, _ = next(iter_source(a.video))
        lane_img = str(HERE / "outputs/lane_frame.jpg")
        imwrite(lane_img, frame)
    cmd = ["lane_extraction.py", "--weights", a.weights, "--image", lane_img,
           "--out", str(docs / "lane_extraction.jpg"), "--device", a.device]
    if a.fx:
        cmd += ["--fx", str(a.fx)]
    run(*cmd)

    readme = HERE / a.readme
    if readme.exists() and (docs / "metrics.md").exists():
        m = json.loads((docs / "metrics.json").read_text())
        block = ("<!-- METRICS_START -->\n"
                 f"Test split, {m['imgsz']} px, `{Path(m['weights']).name}`  "
                 f"(mAP50 **{m['mAP50']:.3f}**, mAP50-95 **{m['mAP50-95']:.3f}**, "
                 f"{m['speed_ms_per_image'].get('inference', '?')} ms/img inference on `{m['device']}`)\n\n"
                 + (docs / "metrics.md").read_text() + "<!-- METRICS_END -->")
        txt = readme.read_text(encoding="utf-8")
        s, e = txt.find("<!-- METRICS_START -->"), txt.find("<!-- METRICS_END -->")
        if s != -1 and e != -1:
            readme.write_text(txt[:s] + block + txt[e + len("<!-- METRICS_END -->"):], encoding="utf-8")
            print(f"README metrics table updated ({readme})")
    print("\nAssets written to docs/. Commit them and your README is ready.")


if __name__ == "__main__":
    main()
