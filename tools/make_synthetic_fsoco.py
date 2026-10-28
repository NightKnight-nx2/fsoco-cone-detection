"""Generate a tiny *synthetic* dataset in FSOCO's Supervisely layout (smoke tests only).

The real FSOCO dataset is ~24 GB and must be downloaded manually. To prove that the
pipeline (data_prep -> train -> evaluate -> inference -> tracking -> lane) works
end-to-end without it, this script renders cartoon cones on a pinhole-camera ground
plane and writes them in exactly the same folder/JSON structure as FSOCO::

    <out>/meta.json
    <out>/<team>/img/<name>.jpg
    <out>/<team>/ann/<name>.jpg.json      # Supervisely: size, objects[classTitle, points.exterior, tags]

It can also render a short forward-driving video through one persistent track, which is
used to test the tracker (cones keep their world position, so IDs should persist).

DO NOT report metrics trained on this data - it is trivially easy and not FSOCO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLASS_NAMES, imwrite  # noqa: E402

W, H = 1280, 720
FX = 900.0
CX, CY = W / 2, H * 0.42          # principal point / horizon row
CAM_H = 0.9                        # camera height (m)
CONE_W = 0.228
BODY_COLORS = {                    # BGR
    0: (200, 70, 20), 1: (30, 210, 240), 2: (20, 120, 250), 3: (20, 90, 250), 4: (80, 160, 70)}


def make_track(rng: np.random.Generator, length: float = 60.0, spacing: float = 3.0,
               width: float = 3.5):
    """Return list of (X, Z, class_id) world cones for one random corner/straight track."""
    k = rng.uniform(-0.004, 0.004)          # curvature: X = k Z^2 (+ slight lateral offset)
    x0 = rng.uniform(-0.4, 0.4)
    cones = []
    z = 3.0
    while z < length:
        xc = x0 + k * z * z
        slope = 2 * k * z
        nrm = np.array([1.0, -slope]) / np.hypot(1.0, slope)   # normal in (X, Z)
        zc = np.array([xc, z])
        left, right = zc - nrm * width / 2, zc + nrm * width / 2
        cones.append((left[0], left[1], 0))    # blue on the left
        cones.append((right[0], right[1], 1))  # yellow on the right
        z += spacing * rng.uniform(0.85, 1.15)
    # orange start gate + a few distractors
    for _ in range(rng.integers(1, 4)):          # orange gates somewhere along the track
        zg = rng.uniform(8, length * 0.9)
        xg = x0 + k * zg * zg
        big = rng.random() < 0.5
        for dx in (-width / 2 - 0.5, width / 2 + 0.5):
            cones.append((xg + dx, zg, 3 if big else 2))
    for _ in range(rng.integers(0, 3)):
        cones.append((x0 + rng.uniform(-6, 6), rng.uniform(6, length * 0.8), 4))
    return cones


def draw_cone(img: np.ndarray, u: float, v: float, hp: float, cls: int, rng: np.random.Generator):
    """Draw a cartoon cone whose base-centre is at (u, v) and pixel height hp."""
    wp = hp * CONE_W / 0.325
    body = BODY_COLORS[cls]
    base_h = max(1, int(hp * 0.07))
    top = (u, v - hp)
    bl, br = (u - wp / 2 * 0.95, v - base_h), (u + wp / 2 * 0.95, v - base_h)
    poly = np.array([top, br, bl], np.int32)
    cv2.fillConvexPoly(img, poly, body, cv2.LINE_AA)
    # stripes
    def band(t0, t1, col):
        def pt(t, s):
            return (u + s * wp / 2 * 0.95 * t, v - base_h - (hp - base_h) * (1 - t))
        p = np.array([pt(t0, -1), pt(t0, 1), pt(t1, 1), pt(t1, -1)], np.int32)
        cv2.fillConvexPoly(img, p, col, cv2.LINE_AA)
    if cls == 0:
        band(0.35, 0.6, (240, 240, 240))
    elif cls == 1:
        band(0.35, 0.6, (25, 25, 25))
    elif cls == 2:
        band(0.35, 0.6, (240, 240, 240))
    elif cls == 3:
        band(0.2, 0.4, (240, 240, 240)); band(0.5, 0.7, (240, 240, 240))
    cv2.rectangle(img, (int(u - wp / 2), int(v - base_h)), (int(u + wp / 2), int(v)),
                  (40, 40, 40), -1)
    return (u - wp / 2, v - hp, u + wp / 2, v)


def background(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    sky = rng.integers(170, 235, 3)
    ground = rng.integers(70, 130, 3)
    g = np.linspace(0, 1, int(CY))[:, None, None]
    img[: int(CY)] = (sky * (0.8 + 0.2 * g)).astype(np.uint8)
    img[int(CY):] = ground
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def render(cones, cam_x: float, cam_z: float, rng: np.random.Generator, bg=None):
    """Render cones seen from (cam_x, cam_z). Returns image + list of (cls, xyxy, world_idx)."""
    img = bg.copy() if bg is not None else background(rng)
    vis = []
    order = sorted(range(len(cones)), key=lambda i: -(cones[i][1] - cam_z))
    for i in order:
        X, Z, c = cones[i]
        z = Z - cam_z
        if z < 3.0 or z > 45:
            continue
        x = X - cam_x
        u = CX + FX * x / z
        v = CY + FX * CAM_H / z
        hp = FX * (0.325 if c != 3 else 0.505) / z
        if hp < 12 or u < 0 or u > W or v - hp > H:
            continue
        box = draw_cone(img, u, v, hp, c, rng)
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
        if (x2 - x1) < 3 or (y2 - y1) < 6:
            continue
        vis.append((c, (float(x1), float(y1), float(x2), float(y2)), i))
    return img, vis


def write_sample(team_dir: Path, name: str, img: np.ndarray, vis) -> None:
    (team_dir / "img").mkdir(parents=True, exist_ok=True)
    (team_dir / "ann").mkdir(parents=True, exist_ok=True)
    imwrite(team_dir / "img" / name, img)
    objs = []
    for c, (x1, y1, x2, y2), _ in vis:
        objs.append({
            "classTitle": CLASS_NAMES[c], "tags": [], "geometryType": "rectangle",
            "points": {"exterior": [[round(x1), round(y1)], [round(x2), round(y2)]], "interior": []},
        })
    ann = {"description": "", "tags": [], "size": {"height": H, "width": W}, "objects": objs}
    (team_dir / "ann" / f"{name}.json").write_text(json.dumps(ann), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic_fsoco", help="output root (FSOCO layout)")
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--video", default=None, help="also render a forward-driving demo video (mp4)")
    ap.add_argument("--video-seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "classes": [{"title": n, "shape": "rectangle"} for n in CLASS_NAMES], "tags": []}))

    n_tracks = max(a.teams, 1) * 3
    tracks = [make_track(rng) for _ in range(n_tracks)]
    for k in range(a.images):
        team = f"synth_team_{k % a.teams:02d}"
        cones = tracks[rng.integers(n_tracks)]
        cam_x, cam_z = rng.uniform(-0.3, 0.3), rng.uniform(0, 25)
        img, vis = render(cones, cam_x, cam_z, rng)
        if len(vis) < 2:
            continue
        write_sample(out / team, f"{team}_{k:05d}.jpg", img, vis)
    print(f"[synthetic] wrote {a.images} images -> {out}")

    if a.video:
        vr = np.random.default_rng(a.seed + 123)
        track = make_track(vr, length=140.0)
        bg = background(vr)
        fps, speed = 30, 6.0
        n = int(a.video_seconds * fps)
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(a.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for f in range(n):
            z = f * speed / fps
            x = 0.15 * np.sin(f / 40)           # slight lateral sway
            img, _ = render(track, x, z, vr, bg)
            vw.write(img)
        vw.release()
        print(f"[synthetic] wrote video -> {a.video}")


if __name__ == "__main__":
    main()
