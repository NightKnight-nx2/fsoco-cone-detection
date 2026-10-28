"""Run the detector on a video, an image, an image folder or a webcam and save an annotated video.

Per-frame detection only (no tracking) - see ``tracking.py`` for persistent IDs.

Examples::

    python inference.py --weights runs/fsoco_yolo11n/weights/best.pt --source drive.mp4 --out outputs/detect.mp4
    python inference.py --weights best.pt --source path/to/frames/ --out outputs/frames.mp4 --fps 15
    python inference.py --weights best.pt --source drive.mp4 --out outputs/detect.mp4 --gif docs/demo.gif
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import OutputSink, draw_detections, iter_source, put_hud, select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file | image | image folder | webcam index")
    ap.add_argument("--out", default="outputs/detections.mp4", help="annotated mp4 ('' to skip)")
    ap.add_argument("--gif", default="", help="optional downscaled GIF for the README")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-seconds", type=float, default=8.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=30.0, help="output fps for image folders (videos keep theirs)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--no-hud", action="store_true", help="hide the FPS / count banner")
    a = ap.parse_args()

    from ultralytics import YOLO

    device = select_device(a.device)
    model = YOLO(a.weights)
    sink = None
    n, t_infer = 0, 0.0
    try:
        for frame, fps in iter_source(a.source, a.fps):
            t0 = time.perf_counter()
            r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=device, verbose=False)[0]
            dt = time.perf_counter() - t0
            t_infer += dt
            b = r.boxes
            vis = draw_detections(frame.copy(), b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                  b.conf.cpu().numpy())
            if not a.no_hud:
                put_hud(vis, f"{1 / max(dt, 1e-6):5.1f} FPS | {len(b)} cones")
            if sink is None:
                sink = OutputSink(a.out or None, a.gif or None, fps, a.gif_width, 12.0, a.gif_seconds)
            sink.write(vis)
            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        if sink is not None:
            sink.close()
    if n:
        print(f"{n} frames, mean model latency {1000 * t_infer / n:.1f} ms ({n / t_infer:.1f} FPS incl. pre/post) on {device}")


if __name__ == "__main__":
    main()
