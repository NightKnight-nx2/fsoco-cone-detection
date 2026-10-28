"""Track cones across a video with Ultralytics' built-in ByteTrack / BoT-SORT (``model.track``).

Adds on top of the raw tracker:

* **Stable class per track** - the displayed class is a confidence-weighted vote over the
  track's history, so a cone doesn't flicker blue <-> unknown frame to frame.
* **Motion trails** of each cone's base point (shows ego-motion through the scene).
* Optional **centre-line overlay** (``--lane``) using the smoothed classes.
* A short summary (unique IDs, mean track length) printed at the end.

Examples::

    python tracking.py --weights best.pt --source drive.mp4 --out outputs/track.mp4 --gif docs/tracking.gif
    python tracking.py --weights best.pt --source drive.mp4 --lane --fx 900
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_COLORS_BGR, OutputSink, draw_detections, iter_source, put_hud, select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file or image-sequence folder")
    ap.add_argument("--out", default="outputs/tracking.mp4")
    ap.add_argument("--gif", default="")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-seconds", type=float, default=8.0)
    ap.add_argument("--tracker", default=str(Path(__file__).parent / "configs" / "bytetrack_cones.yaml"),
                    help="tracker yaml; use 'botsort.yaml' for BoT-SORT")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=30.0, help="output fps for image folders")
    ap.add_argument("--trail", type=int, default=25, help="trail length in frames (0 = off)")
    ap.add_argument("--lane", action="store_true", help="overlay the estimated centre line")
    ap.add_argument("--fx", type=float, default=None, help="focal length px for metric lane mode")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO

    from lane_extraction import draw_lane, estimate_centerline

    model = YOLO(a.weights)
    device = select_device(a.device)
    src = a.source
    fps = next(iter_source(src, a.fps))[1]  # peek fps (cheap for videos)

    votes: dict[int, Counter] = defaultdict(Counter)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(a.trail, 1)))
    first_seen, last_seen = {}, {}
    sink = OutputSink(a.out or None, a.gif or None, fps, a.gif_width, 12.0, a.gif_seconds)
    n = 0
    stream = model.track(source=src, stream=True, persist=True, tracker=a.tracker, imgsz=a.imgsz,
                         conf=a.conf, iou=a.iou, device=device, verbose=False)
    try:
        for r in stream:
            frame = r.orig_img.copy()
            b = r.boxes
            if b is not None and b.id is not None and len(b):
                xyxy = b.xyxy.cpu().numpy()
                ids = b.id.cpu().numpy().astype(int)
                conf = b.conf.cpu().numpy()
                cls_raw = b.cls.cpu().numpy().astype(int)
                cls = np.empty_like(cls_raw)
                for i, (tid, c, s) in enumerate(zip(ids, cls_raw, conf)):
                    votes[tid][c] += float(s)
                    cls[i] = votes[tid].most_common(1)[0][0]
                    first_seen.setdefault(tid, n)
                    last_seen[tid] = n
                    trails[tid].append((int((xyxy[i, 0] + xyxy[i, 2]) / 2), int(xyxy[i, 3])))
                if a.trail:
                    for i, tid in enumerate(ids):
                        pts = list(trails[tid])
                        for k in range(1, len(pts)):
                            alpha = k / len(pts)
                            col = tuple(int(c * (0.35 + 0.65 * alpha)) for c in CLASS_COLORS_BGR[cls[i]])
                            cv2.line(frame, pts[k - 1], pts[k], col, 2, cv2.LINE_AA)
                if a.lane:
                    h, w = frame.shape[:2]
                    frame = draw_lane(frame, estimate_centerline(xyxy, cls, (w, h), fx=a.fx))
                draw_detections(frame, xyxy, cls, conf, ids)
                put_hud(frame, f"frame {n} | {len(ids)} tracked | {len(first_seen)} unique IDs")
            sink.write(frame)
            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        sink.close()

    lengths = [last_seen[t] - first_seen[t] + 1 for t in first_seen]
    if lengths:
        print(f"{n} frames | {len(lengths)} unique track IDs | mean track length {np.mean(lengths):.1f} frames "
              f"| tracks >= 30 frames: {sum(l >= 30 for l in lengths)}")


if __name__ == "__main__":
    main()
