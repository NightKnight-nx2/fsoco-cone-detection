"""Shared helpers for the FSOCO cone-detection pipeline.

Holds the class definition (single source of truth), Unicode-safe image I/O,
drawing utilities, video/GIF sinks and small dataset helpers used by every
other script in the repository.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# FSOCO bounding-box class titles (Supervisely ``classTitle``) in the order the
# model uses. NOTE: FSOCO's website calls them "Blue / Yellow / Small orange /
# Large orange / Other"; the annotation files use the snake_case titles below.
CLASS_NAMES: list[str] = [
    "blue_cone",
    "yellow_cone",
    "orange_cone",
    "large_orange_cone",
    "unknown_cone",
]
NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}

# BGR colours used for drawing (approximate real cone colours).
CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (235, 110, 30),   # blue
    (0, 215, 255),    # yellow
    (0, 140, 255),    # orange
    (50, 50, 240),    # large orange (drawn reddish to tell it apart)
    (90, 200, 80),    # unknown
]

# Physical cone heights in metres (FSG rules: small 325 mm, large 505 mm).
CONE_HEIGHT_M: dict[int, float] = {0: 0.325, 1: 0.325, 2: 0.325, 3: 0.505, 4: 0.325}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


# --------------------------------------------------------------------------- #
# Unicode-safe image I/O (cv2.imread/imwrite fail on non-ASCII Windows paths,
# e.g. "C:/Users/<name>/Masaüstü/...").
# --------------------------------------------------------------------------- #
def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except OSError:
        return None


def imwrite(path: str | Path, img: np.ndarray, quality: int = 95) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def select_device(device: str = "auto"):
    """Return an Ultralytics-compatible device string/int for ``device``."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_detections(
    frame: np.ndarray,
    xyxy: np.ndarray,
    cls: Sequence[int],
    conf: Optional[Sequence[float]] = None,
    ids: Optional[Sequence[int]] = None,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Draw clean class-coloured boxes with ``[#id] class [conf]`` labels in place."""
    h, w = frame.shape[:2]
    th = thickness or max(1, round(min(h, w) / 400))
    font_scale = max(0.35, min(h, w) / 1400)
    for i, (box, c) in enumerate(zip(np.asarray(xyxy), cls)):
        c = int(c)
        color = CLASS_COLORS_BGR[c % len(CLASS_COLORS_BGR)]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
        parts = []
        if ids is not None:
            parts.append(f"#{int(ids[i])}")
        parts.append(CLASS_NAMES[c % len(CLASS_NAMES)].replace("_cone", ""))
        if conf is not None:
            parts.append(f"{float(conf[i]):.2f}")
        label = " ".join(parts)
        (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 - 2 if y1 - tht - bl - 2 > 0 else y1 + tht + bl + 2
        cv2.rectangle(frame, (x1, ty - tht - bl), (x1 + tw + 4, ty + 1), color, -1)
        # black or white text depending on box colour brightness
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        txt = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 2, ty - bl // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, txt, 1, cv2.LINE_AA)
    return frame


def put_hud(frame: np.ndarray, text: str) -> np.ndarray:
    """Small translucent text banner in the top-left corner."""
    h, w = frame.shape[:2]
    scale = max(0.5, min(h, w) / 900)
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (16 + tw, 16 + th + bl), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Sources and sinks
# --------------------------------------------------------------------------- #
def iter_source(source: str, default_fps: float = 30.0) -> Iterator[tuple[np.ndarray, float]]:
    """Yield ``(bgr_frame, fps)`` from a video file, webcam index, image or image folder."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        p = Path(source)
        if p.is_dir() or p.suffix.lower() in IMG_EXTS:
            files = sorted(f for f in (p.iterdir() if p.is_dir() else [p])
                           if f.suffix.lower() in IMG_EXTS)
            if not files:
                raise FileNotFoundError(f"No images found in {source}")
            for f in files:
                img = imread(f)
                if img is not None:
                    yield img, default_fps
            return
        if not p.exists():
            raise FileNotFoundError(source)
        cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, fps
    finally:
        cap.release()


class OutputSink:
    """Writes frames to an MP4 and/or a downscaled GIF.

    The MP4 is first written with OpenCV's ``mp4v`` codec and, when ``ffmpeg``
    is on PATH, re-encoded to H.264/yuv420p so it plays everywhere (browsers,
    GitHub, QuickTime).
    """

    def __init__(self, video_path: Optional[str], gif_path: Optional[str], fps: float,
                 gif_width: int = 640, gif_fps: float = 12.0, gif_max_seconds: float = 10.0):
        self.video_path = Path(video_path) if video_path else None
        self.gif_path = Path(gif_path) if gif_path else None
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._raw_path: Optional[Path] = None
        self._gif_frames: list = []
        self._gif_width = gif_width
        self._gif_step = max(1, round(fps / gif_fps))
        self._gif_fps = fps / self._gif_step
        self._gif_max = int(gif_max_seconds * self._gif_fps)
        self._n = 0

    def write(self, frame: np.ndarray) -> None:
        if self.video_path is not None:
            if self._writer is None:
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                self._raw_path = self.video_path.with_suffix(".raw.mp4")
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self._raw_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._writer.write(frame)
        if self.gif_path is not None and self._n % self._gif_step == 0 \
                and len(self._gif_frames) < self._gif_max:
            h, w = frame.shape[:2]
            s = self._gif_width / w
            small = cv2.resize(frame, (self._gif_width, int(h * s)), interpolation=cv2.INTER_AREA)
            self._gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self._n += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            if shutil.which("ffmpeg"):
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self._raw_path),
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       str(self.video_path)]
                if subprocess.run(cmd).returncode == 0:
                    self._raw_path.unlink(missing_ok=True)
                else:
                    self._raw_path.replace(self.video_path)
            else:
                self._raw_path.replace(self.video_path)
            print(f"[sink] video  -> {self.video_path}")
        if self.gif_path is not None and self._gif_frames:
            from PIL import Image

            self.gif_path.parent.mkdir(parents=True, exist_ok=True)
            imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.FASTOCTREE)
                    for f in self._gif_frames]
            imgs[0].save(self.gif_path, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / self._gif_fps), loop=0, optimize=True)
            print(f"[sink] gif    -> {self.gif_path} ({len(imgs)} frames)")


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def read_yolo_labels(label_path: str | Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO txt label file -> (xyxy pixels (N,4), class ids (N,))."""
    p = Path(label_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), int)
    arr = np.loadtxt(p, ndmin=2, dtype=np.float64)
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return xyxy.astype(np.float32), cls


def label_path_for(image_path: str | Path) -> Path:
    """Ultralytics convention: ``.../images/<split>/x.jpg`` -> ``.../labels/<split>/x.txt``."""
    p = Path(image_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def resolve_split_images(data_yaml: str | Path, split: str) -> list[Path]:
    """List the image files of ``split`` ('train'/'val'/'test') from an Ultralytics data.yaml."""
    import yaml

    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entry = cfg.get(split)
    if entry is None:
        raise KeyError(f"'{split}' not defined in {data_yaml}")
    entry_p = Path(entry) if Path(entry).is_absolute() else root / entry
    if entry_p.is_dir():
        files = [f for f in entry_p.rglob("*") if f.suffix.lower() in IMG_EXTS]
    else:  # a txt list file (may contain duplicates from oversampling)
        files = [Path(line.strip()) for line in entry_p.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        files = list(dict.fromkeys(files))
    return sorted(files)
