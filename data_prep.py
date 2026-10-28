"""Convert the FSOCO bounding-box dataset (Supervisely format) to Ultralytics YOLO format.

FSOCO ships one folder per team::

    <src>/<team>/img/<name>.jpg
    <src>/<team>/ann/<name>.jpg.json   # {"size": {"width","height"}, "objects": [{"classTitle",
                                       #   "points": {"exterior": [[x1,y1],[x2,y2]]}, "tags": [...]}]}

and writes the layout Ultralytics expects::

    <out>/images/{train,val,test}/<team>__<name>.jpg
    <out>/labels/{train,val,test}/<team>__<name>.txt   # "cls cx cy w h" normalised
    <out>/data.yaml, <out>/stats.json, <out>/class_distribution.png, <out>/preview/*.jpg

Class ids follow ``common.CLASS_NAMES``. Segmentation classes (``seg_*``) are ignored.

Class imbalance: ``--oversample`` enables LVIS-style *repeat-factor sampling* - training images
that contain rare classes are listed several times in ``train.txt`` (Ultralytics has no
per-class loss weights, so this is the cleanest way to re-balance).

Examples::

    python data_prep.py --src D:/datasets/fsoco_bounding_boxes_train --out data/fsoco_yolo
    python data_prep.py --src ... --out data/fsoco_small --subset 400      # quick smoke subset
    python data_prep.py --src ... --out data/fsoco_yolo --oversample --split-by team
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES, NAME_TO_ID, draw_detections, imread, imwrite, read_yolo_labels

# FSOCO images exported through the Supervisely importer can carry a 140 px watermark border.
# If your images have it (labels would then look shifted in the preview) pass --remove-watermark.
WATERMARK_BORDER = 140


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def find_annotations(src: Path) -> list[Path]:
    return sorted(src.glob("**/ann/*.json"))


def team_of(ann: Path, src: Path) -> str:
    rel = ann.relative_to(src)
    return rel.parts[0] if len(rel.parts) > 2 else src.name


def image_for(ann: Path) -> Path:
    return ann.parent.parent / "img" / ann.name[: -len(".json")]


def place_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:  # e.g. Windows without developer mode -> fall back to copy
            pass
    elif mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def convert_one(job: tuple) -> dict:
    """Convert a single annotated image. Runs in worker processes."""
    ann_path, team, split, out_dir, o = job
    ann_path, out_dir = Path(ann_path), Path(out_dir)
    res = {"status": "ok", "split": split, "classes": Counter(), "skipped": Counter(), "name": None}
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        res["status"] = "bad_json"
        return res
    img_path = image_for(ann_path)
    if not img_path.exists():
        res["status"] = "missing_image"
        return res

    iw, ih = data["size"]["width"], data["size"]["height"]
    b = WATERMARK_BORDER if o["remove_watermark"] else 0
    w, h = iw - 2 * b, ih - 2 * b
    lines: list[str] = []
    for obj in data.get("objects", []):
        title = obj.get("classTitle")
        if title not in NAME_TO_ID:  # seg_* classes, typos, ...
            res["skipped"][f"class:{title}"] += 1
            continue
        tags = {t["name"] for t in obj.get("tags", [])}
        if tags & set(o["exclude_tags"]):
            res["skipped"]["excluded_by_tag"] += 1
            continue
        pts = obj.get("points", {}).get("exterior")
        if not pts or len(pts) < 2:
            res["skipped"]["no_geometry"] += 1
            continue
        (xa, ya), (xb, yb) = pts[0], pts[1]
        x1, x2 = sorted((xa - b, xb - b))
        y1, y2 = sorted((ya - b, yb - b))
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if (x2 - x1) < o["min_box_px"] or (y2 - y1) < o["min_box_px"]:
            res["skipped"]["tiny_or_degenerate"] += 1
            continue
        cid = NAME_TO_ID[title]
        lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        res["classes"][cid] += 1
    if not lines and not o["keep_empty"]:
        res["status"] = "no_labels"
        return res

    need_decode = b > 0 or o["max_side"] > 0
    ext = ".jpg" if need_decode else img_path.suffix.lower()
    name = sanitize(f"{team}__{img_path.stem}") + ext
    dst_img = out_dir / "images" / split / name
    if need_decode:
        img = imread(img_path)
        if img is None:
            res["status"] = "unreadable_image"
            return res
        if b:
            img = img[b:-b, b:-b]
        m = max(img.shape[:2])
        if o["max_side"] > 0 and m > o["max_side"]:
            s = o["max_side"] / m
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        imwrite(dst_img, img)
    else:
        place_image(img_path, dst_img, o["link"])
    lbl = out_dir / "labels" / split / (Path(name).stem + ".txt")
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    res["name"] = str(dst_img)
    return res


def assign_splits(anns: list[Path], src: Path, val_frac: float, test_frac: float,
                  mode: str, seed: int) -> dict[Path, str]:
    rng = random.Random(seed)
    teams = defaultdict(list)
    for a in anns:
        teams[team_of(a, src)].append(a)
    if mode == "auto":
        mode = "team" if len(teams) >= 8 else "image"
    out: dict[Path, str] = {}
    if mode == "image":
        for a in anns:
            r = rng.random()
            out[a] = "test" if r < test_frac else "val" if r < test_frac + val_frac else "train"
    else:  # whole teams go to one split -> no near-duplicate frames leaking across splits
        names = sorted(teams)
        rng.shuffle(names)
        total = len(anns)
        acc = {"test": 0, "val": 0}
        for t in names:
            n = len(teams[t])
            if acc["test"] < test_frac * total:
                s = "test"
            elif acc["val"] < val_frac * total:
                s = "val"
            else:
                s = "train"
            if s in acc:
                acc[s] += n
            for a in teams[t]:
                out[a] = s
    print(f"[split] mode={mode}  teams={len(teams)}")
    if mode == "image":
        print("        WARNING: consecutive video frames are near-duplicates; random image-level "
              "splits leak and inflate mAP. Prefer --split-by team for honest numbers.")
    return out


def repeat_factors(train_labels: dict[str, set[int]], t: float, seed: int) -> dict[str, int]:
    """LVIS repeat-factor sampling: r_c = max(1, sqrt(t / f_c)); image r = max over its classes."""
    n = max(len(train_labels), 1)
    freq = Counter(c for cs in train_labels.values() for c in cs)
    r_c = {c: max(1.0, math.sqrt(t / (freq[c] / n))) for c in freq}
    rng = random.Random(seed)
    reps = {}
    for img, cs in train_labels.items():
        r = max((r_c[c] for c in cs), default=1.0)
        reps[img] = int(r) + (1 if rng.random() < (r - int(r)) else 0)
    print("[oversample] class repeat factors:",
          {CLASS_NAMES[c]: round(v, 2) for c, v in sorted(r_c.items())})
    return reps


def plot_distribution(stats: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import CLASS_COLORS_BGR

    splits = [s for s in ("train", "val", "test") if stats["boxes"].get(s)]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(splits), 1)
    for i, s in enumerate(splits):
        vals = [stats["boxes"][s].get(n, 0) for n in CLASS_NAMES]
        ax.bar(np.arange(len(CLASS_NAMES)) + i * width, vals, width, label=s)
    ax.set_xticks(np.arange(len(CLASS_NAMES)) + width * (len(splits) - 1) / 2)
    ax.set_xticklabels([n.replace("_cone", "") for n in CLASS_NAMES])
    ax.set_yscale("log")
    ax.set_ylabel("boxes (log scale)")
    ax.set_title("FSOCO class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_previews(out: Path, n: int, seed: int) -> None:
    """Draw converted labels on a few training images - quick visual sanity check."""
    files = sorted((out / "images" / "train").glob("*"))
    rng = random.Random(seed)
    for f in rng.sample(files, min(n, len(files))):
        img = imread(f)
        if img is None:
            continue
        lp = out / "labels" / "train" / (f.stem + ".txt")
        xyxy, cls = read_yolo_labels(lp, img.shape[1], img.shape[0])
        imwrite(out / "preview" / f"{f.stem}.jpg", draw_detections(img, xyxy, cls))
    print(f"[preview] wrote {min(n, len(files))} label overlays -> {out / 'preview'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="extracted FSOCO bounding-box folder (contains team folders)")
    ap.add_argument("--out", default="data/fsoco_yolo", help="output dataset folder")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-by", choices=["auto", "team", "image"], default="auto",
                    help="'team' keeps all frames of a team in one split (honest); 'image' is random")
    ap.add_argument("--subset", type=int, default=0, help="use only N random annotated images (smoke test)")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="downscale images so the longer side is <= this (0 = keep original, then linked)")
    ap.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink",
                    help="how to place images when no resize is needed")
    ap.add_argument("--exclude-tags", nargs="*", default=[],
                    help="drop boxes carrying these tags, e.g. knocked_over truncated")
    ap.add_argument("--min-box-px", type=float, default=2.0, help="drop boxes smaller than this (px)")
    ap.add_argument("--keep-empty", action="store_true", help="keep images without any usable box")
    ap.add_argument("--remove-watermark", action="store_true", help=f"crop a {WATERMARK_BORDER}px border")
    ap.add_argument("--oversample", action="store_true", help="repeat-factor sampling for rare classes")
    ap.add_argument("--oversample-t", type=float, default=0.5,
                    help="repeat-factor threshold t (higher = more aggressive)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--preview", type=int, default=8, help="number of label-overlay previews to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="delete --out before writing")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    anns = find_annotations(src)
    if not anns:
        raise SystemExit(f"No '*/ann/*.json' files under {src}. Is --src the extracted FSOCO folder?")
    print(f"[scan] {len(anns)} annotation files in {src}")
    if a.subset and a.subset < len(anns):
        anns = sorted(random.Random(a.seed).sample(anns, a.subset))
        print(f"[scan] using random subset of {len(anns)}")
    if a.clean and out.exists():
        shutil.rmtree(out)

    split_of = assign_splits(anns, src, a.val_frac, a.test_frac, a.split_by, a.seed)
    opts = dict(remove_watermark=a.remove_watermark, exclude_tags=a.exclude_tags,
                min_box_px=a.min_box_px, keep_empty=a.keep_empty, max_side=a.max_side, link=a.link)
    jobs = [(str(p), team_of(p, src), split_of[p], str(out), opts) for p in anns]

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    results = []
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for r in tqdm(pool.imap_unordered(convert_one, jobs, chunksize=16), total=len(jobs), desc="convert"):
                results.append(r)
    else:
        for j in tqdm(jobs, desc="convert"):
            results.append(convert_one(j))

    status = Counter(r["status"] for r in results)
    boxes = {s: Counter() for s in ("train", "val", "test")}
    images = Counter()
    skipped = Counter()
    train_classes: dict[str, set[int]] = {}
    for r in results:
        skipped.update(r["skipped"])
        if r["status"] != "ok":
            continue
        images[r["split"]] += 1
        for c, n in r["classes"].items():
            boxes[r["split"]][CLASS_NAMES[c]] += n
        if r["split"] == "train":
            train_classes[r["name"]] = set(r["classes"])

    # --- data.yaml (+ optional oversampled train list) -------------------------------------- #
    out = out.resolve()
    train_entry = "images/train"
    if a.oversample and train_classes:
        reps = repeat_factors(train_classes, a.oversample_t, a.seed)
        lines = [Path(p).resolve().as_posix() for p, k in sorted(reps.items()) for _ in range(k)]
        (out / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        train_entry = "train.txt"
        print(f"[oversample] train list: {len(train_classes)} unique -> {len(lines)} entries")
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(
        f"# Auto-generated by data_prep.py\npath: {out.as_posix()}\n"
        f"train: {train_entry}\nval: images/val\ntest: images/test\nnames:\n{names_block}\n",
        encoding="utf-8")

    stats = {"status": dict(status), "images": dict(images),
             "boxes": {s: dict(c) for s, c in boxes.items()}, "skipped": dict(skipped),
             "split_by": a.split_by, "oversample": a.oversample, "args": vars(a)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_distribution(stats, out / "class_distribution.png")
    if a.preview:
        write_previews(out, a.preview, a.seed)

    print("\n=== summary ===")
    print("status :", dict(status))
    print("images :", dict(images))
    print(f"{'class':<20}" + "".join(f"{s:>9}" for s in ("train", "val", "test")))
    for n in CLASS_NAMES:
        print(f"{n:<20}" + "".join(f"{boxes[s][n]:>9}" for s in ("train", "val", "test")))
    if skipped:
        print("skipped:", dict(skipped))
    print(f"\ndata.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
