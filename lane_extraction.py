"""Estimate a track centre line from detected blue (left) and yellow (right) cones.

Formula Student rule: blue cones mark the LEFT boundary, yellow the RIGHT boundary
(driving direction). Orange cones are start/finish markers and ignored here.

Two modes
---------
* **Image space** (default): each boundary is fitted as ``x = f(y)`` through the cone *base*
  points (bottom-centre of each box); the centre line is the mean of the two curves on
  their common ``y`` range. Purely 2D, no calibration.
* **Approximate metric / bird's-eye** (``--fx <focal px>``): with a pinhole model and the known
  cone height (325 mm small, 505 mm large) the depth of each cone is ``Z = fx * H / h_px`` and
  lateral position ``X = (u - cx) * Z / fx``. Boundaries are fitted as ``X = f(Z)`` and the
  centre line is drawn both top-down and re-projected into the image. If only one side is
  visible the centre is offset by half the track width. Depth accuracy depends on the focal
  length you provide and on boxes covering the full cone height - treat it as an estimate.

Fit methods: ``poly`` (robust low-degree polynomial, default) or ``spline`` (smoothing spline).

Examples::

    python lane_extraction.py --weights best.pt --image frame.jpg --out docs/lane.jpg
    python lane_extraction.py --weights best.pt --image frame.jpg --out docs/lane_bev.jpg --fx 900
    python lane_extraction.py --labels data/fsoco_yolo/labels/test/x.txt --image data/.../x.jpg --out lane_gt.jpg
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common import (CLASS_COLORS_BGR, CONE_HEIGHT_M, draw_detections, imread, imwrite,
                    read_yolo_labels, select_device)

LEFT_CLS, RIGHT_CLS = 0, 1  # blue, yellow


@dataclass
class LaneResult:
    mode: str                                   # "image" | "bev"
    left_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    right_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    left_curve: Optional[np.ndarray] = None     # (M,2) in image px (mode image) or (X,Z) metres (bev)
    right_curve: Optional[np.ndarray] = None
    center_curve: Optional[np.ndarray] = None
    center_img: Optional[np.ndarray] = None     # centre line in image pixels (both modes)
    left_img: Optional[np.ndarray] = None
    right_img: Optional[np.ndarray] = None
    note: str = ""


def _fit_curve(t: np.ndarray, x: np.ndarray, method: str, degree: int, smooth: float):
    """Fit x = f(t) and return a callable (or None if impossible)."""
    n = len(t)
    if n < 2:
        return None
    order = np.argsort(t)
    t, x = t[order], x[order]
    if method == "spline" and n >= 4:
        from scipy.interpolate import UnivariateSpline

        # merge duplicate t values (spline needs strictly increasing)
        tu, inv = np.unique(np.round(t, 6), return_inverse=True)
        xu = np.bincount(inv, weights=x) / np.bincount(inv)
        if len(tu) >= 4:
            sp = UnivariateSpline(tu, xu, k=min(3, len(tu) - 1), s=len(tu) * smooth ** 2)
            return sp
    deg = min(degree, n - 1)
    coef = np.polyfit(t, x, deg)
    return np.poly1d(coef)


def _range(a: np.ndarray, b: np.ndarray, margin: float):
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    if hi - lo < 1e-6:                               # no overlap -> use union
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    return lo - margin, hi + margin


def estimate_centerline(xyxy: np.ndarray, cls: np.ndarray, img_size: tuple[int, int],
                        method: str = "poly", degree: int = 2, fx: Optional[float] = None,
                        track_width: float = 3.5, n_samples: int = 60, smooth: float = 2.0
                        ) -> LaneResult:
    """Compute left/right boundary curves and the centre line from cone detections.

    Args:
        xyxy: (N,4) boxes in pixels. cls: (N,) class ids. img_size: (width, height).
        fx: focal length in pixels -> enables the metric/bird's-eye mode.
    """
    w, h = img_size
    xyxy, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(cls, int)
    base = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)   # (u, v) cone base point
    boxh = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)
    is_l, is_r = cls == LEFT_CLS, cls == RIGHT_CLS

    if fx is None:
        res = LaneResult("image", left_pts=base[is_l], right_pts=base[is_r])
        fl = _fit_curve(base[is_l, 1], base[is_l, 0], method, degree, smooth)
        fr = _fit_curve(base[is_r, 1], base[is_r, 0], method, degree, smooth)
        if fl is None or fr is None:
            res.note = "need >= 2 blue and >= 2 yellow cones"
            for f, pts, name in ((fl, base[is_l], "left_curve"), (fr, base[is_r], "right_curve")):
                if f is not None:
                    ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n_samples)
                    setattr(res, name, np.stack([f(ys), ys], 1))
            return res
        ys_l = np.linspace(base[is_l, 1].min(), base[is_l, 1].max(), n_samples)
        ys_r = np.linspace(base[is_r, 1].min(), base[is_r, 1].max(), n_samples)
        res.left_curve = np.stack([fl(ys_l), ys_l], 1)
        res.right_curve = np.stack([fr(ys_r), ys_r], 1)
        lo, hi = _range(base[is_l, 1], base[is_r, 1], margin=0.02 * h)
        ys = np.linspace(lo, hi, n_samples)
        res.center_curve = np.stack([(fl(ys) + fr(ys)) / 2, ys], 1)
        res.center_img, res.left_img, res.right_img = res.center_curve, res.left_curve, res.right_curve
        return res

    # ---------------- metric mode ----------------
    cx = w / 2
    heights = np.array([CONE_HEIGHT_M.get(int(c), 0.325) for c in cls])
    Z = fx * heights / boxh
    X = (base[:, 0] - cx) * Z / fx
    pts = np.stack([X, Z], 1)
    res = LaneResult("bev", left_pts=pts[is_l], right_pts=pts[is_r])
    # row model v = a + b / Z  (ground plane seen by a level camera) fitted on all cone bases
    A = np.stack([np.ones_like(Z), 1.0 / Z], 1)
    (a, b), *_ = np.linalg.lstsq(A, base[:, 1], rcond=None) if len(Z) >= 2 else ((h * 0.42, fx * 0.9), None)

    def to_img(curve):
        if curve is None:
            return None
        Xc, Zc = curve[:, 0], np.maximum(curve[:, 1], 0.5)
        return np.stack([cx + fx * Xc / Zc, a + b / Zc], 1)

    fl = _fit_curve(pts[is_l, 1], pts[is_l, 0], method, degree, smooth)
    fr = _fit_curve(pts[is_r, 1], pts[is_r, 0], method, degree, smooth)
    if fl is None and fr is None:
        res.note = "need >= 2 blue or >= 2 yellow cones"
        return res
    zs_l = np.linspace(pts[is_l, 1].min(), pts[is_l, 1].max(), n_samples) if fl else None
    zs_r = np.linspace(pts[is_r, 1].min(), pts[is_r, 1].max(), n_samples) if fr else None
    if fl:
        res.left_curve = np.stack([fl(zs_l), zs_l], 1)
    if fr:
        res.right_curve = np.stack([fr(zs_r), zs_r], 1)
    if fl and fr:
        lo, hi = _range(pts[is_l, 1], pts[is_r, 1], margin=0.5)
        zs = np.linspace(max(lo, 0.5), hi, n_samples)
        res.center_curve = np.stack([(fl(zs) + fr(zs)) / 2, zs], 1)
    else:                                            # one side only -> offset by half track width
        side, zs, sign = (fl, zs_l, +1) if fl else (fr, zs_r, -1)
        res.center_curve = np.stack([side(zs) + sign * track_width / 2, zs], 1)
        res.note = "single side visible: centre offset by half track width"
    res.left_img, res.right_img, res.center_img = (to_img(res.left_curve), to_img(res.right_curve),
                                                   to_img(res.center_curve))
    return res


def _poly(img, pts, color, th):
    if pts is not None and len(pts) >= 2:
        cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)


def draw_lane(frame: np.ndarray, res: LaneResult, bev: bool = True) -> np.ndarray:
    """Overlay boundaries + centre line (and a bird's-eye inset in metric mode)."""
    th = max(2, frame.shape[0] // 240)
    _poly(frame, res.left_img, CLASS_COLORS_BGR[0], th)
    _poly(frame, res.right_img, CLASS_COLORS_BGR[1], th)
    _poly(frame, res.center_img, (60, 220, 60), th + 2)
    if res.mode == "bev" and bev and res.center_curve is not None:
        frame = _draw_bev_inset(frame, res)
    if res.note:
        cv2.putText(frame, res.note, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _draw_bev_inset(frame: np.ndarray, res: LaneResult, size: tuple[int, int] = (260, 340)) -> np.ndarray:
    pw, ph = size
    canvas = np.full((ph, pw, 3), 30, np.uint8)
    allpts = np.vstack([p for p in (res.left_pts, res.right_pts, res.center_curve) if p is not None and len(p)])
    xmax = max(3.0, np.abs(allpts[:, 0]).max() * 1.15)
    zmax = max(8.0, allpts[:, 1].max() * 1.1)

    def m2p(p):
        return np.stack([pw / 2 + p[:, 0] / xmax * (pw / 2 - 8), ph - 14 - p[:, 1] / zmax * (ph - 28)], 1)

    for z in range(0, int(zmax) + 1, 5):
        y = int(m2p(np.array([[0.0, z]]))[0, 1])
        cv2.line(canvas, (0, y), (pw, y), (55, 55, 55), 1)
        cv2.putText(canvas, f"{z}m", (3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for pts, col in ((res.left_pts, CLASS_COLORS_BGR[0]), (res.right_pts, CLASS_COLORS_BGR[1])):
        for p in m2p(pts) if len(pts) else []:
            cv2.circle(canvas, tuple(int(v) for v in p), 4, col, -1, cv2.LINE_AA)
    _poly(canvas, m2p(res.center_curve), (60, 220, 60), 2)
    cv2.circle(canvas, (pw // 2, ph - 14), 5, (255, 255, 255), -1)     # ego vehicle
    cv2.putText(canvas, "top-down (approx.)", (6, ph - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    h, w = frame.shape[:2]
    frame[10:10 + ph, w - pw - 10:w - 10] = canvas
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", help="detector weights (omit when using --labels)")
    ap.add_argument("--labels", help="use a YOLO label .txt (ground truth) instead of a detector")
    ap.add_argument("--out", default="docs/lane_extraction.jpg")
    ap.add_argument("--method", choices=["poly", "spline"], default="poly")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--fx", type=float, default=None, help="focal length in px -> metric / top-down mode")
    ap.add_argument("--track-width", type=float, default=3.5)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    img = imread(a.image)
    if img is None:
        raise SystemExit(f"Cannot read {a.image}")
    h, w = img.shape[:2]
    if a.labels:
        xyxy, cls = read_yolo_labels(a.labels, w, h)
        conf = None
    elif a.weights:
        from ultralytics import YOLO

        r = YOLO(a.weights).predict(img, imgsz=a.imgsz, conf=a.conf, device=select_device(a.device), verbose=False)[0]
        xyxy, cls, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
    else:
        raise SystemExit("Provide --weights or --labels")

    res = estimate_centerline(xyxy, cls, (w, h), a.method, a.degree, a.fx, a.track_width)
    vis = draw_detections(img.copy(), xyxy, cls, conf)
    vis = draw_lane(vis, res)
    imwrite(a.out, vis)
    print(f"mode={res.mode}  left={len(res.left_pts)} right={len(res.right_pts)}  note='{res.note}'  -> {a.out}")


if __name__ == "__main__":
    main()
