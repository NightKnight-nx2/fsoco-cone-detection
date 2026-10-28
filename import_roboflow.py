"""Import a Roboflow/YOLOv8-format export of a cone dataset into this project's canonical layout.

Why not use the export as is?
  * Roboflow orders classes alphabetically (blue, large_orange, orange, unknown, yellow), but the
    rest of this repo (drawing colours, centre-line extraction: blue = left, yellow = right) relies on
    the order in ``common.CLASS_NAMES``. This script **remaps class ids by name**.
  * Roboflow's train/valid/test split is random per image. Frames of one team come from the same
    driving sequences (near-duplicates), so a random split leaks and inflates mAP. By default we
    re-split **by team** (team = filename prefix, e.g. ``amz_00012_jpg.rf.<hash>.jpg`` -> ``amz``).

The output is identical to ``data_prep.py`` (images/, labels/, data.yaml, stats.json, preview/ ...).

Example::

    python import_roboflow.py --src data/fsoco_mono.v1i.yolov8 --out data/fsoco_mono_yolo --oversample
    python import_roboflow.py --src ... --split-by original      # keep Roboflow's own split
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from common import CLASS_NAMES, IMG_EXTS
from data_prep import plot_distribution, repeat_factors, write_previews

ROBOFLOW_SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def team_of(name: str) -> str:
    m = re.match(r"^(.*?)_\d", name)
    return (m.group(1) if m else "unknown_team").lower()


def clean_stem(name: str) -> str:
    """'amz_00000_jpg.rf.<hash>.jpg' -> 'amz_00000'"""
    stem = Path(name).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    return re.sub(r"_(jpg|jpeg|png)$", "", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Roboflow export folder (contains data.yaml, train/, valid/, test/)")
    ap.add_argument("--out", default="data/fsoco_mono_yolo")
    ap.add_argument("--split-by", choices=["team", "original"], default="team")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--oversample", action="store_true", help="repeat-factor sampling for rare classes")
    ap.add_argument("--oversample-t", type=float, default=0.5)
    ap.add_argument("--preview", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    unknown = [n for n in names if n not in CLASS_NAMES]
    if unknown:
        raise SystemExit(f"Classes {unknown} are not in common.CLASS_NAMES={CLASS_NAMES}. Edit common.py first.")
    remap = {i: CLASS_NAMES.index(n) for i, n in enumerate(names)}
    print("[classes] roboflow order ->", {i: f"{n} -> {remap[i]}" for i, n in enumerate(names)})

    items = []  # (image_path, label_path, original_split)
    for rf_split, ours in ROBOFLOW_SPLITS.items():
        d = src / rf_split
        if not (d / "images").is_dir():
            continue
        for img in sorted((d / "images").iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                items.append((img, d / "labels" / (img.stem + ".txt"), ours))
    if not items:
        raise SystemExit(f"No images found under {src}")
    print(f"[scan] {len(items)} images")

    # --- decide splits -------------------------------------------------------------------- #
    if a.split_by == "original":
        split_of = {i: it[2] for i, it in enumerate(items)}
    else:
        by_team = defaultdict(list)
        for i, it in enumerate(items):
            by_team[team_of(it[0].name)].append(i)
        teams = sorted(by_team)
        random.Random(a.seed).shuffle(teams)
        acc = {"test": 0, "val": 0}
        split_of = {}
        for t in teams:
            n = len(by_team[t])
            s = ("test" if acc["test"] < a.test_frac * len(items)
                 else "val" if acc["val"] < a.val_frac * len(items) else "train")
            if s in acc:
                acc[s] += n
            for i in by_team[t]:
                split_of[i] = s
        print(f"[split] by team ({len(teams)} teams):",
              {s: sorted({team_of(items[i][0].name) for i in split_of if split_of[i] == s}) for s in ("val", "test")})

    if a.clean and out.exists():
        shutil.rmtree(out)
    boxes = {s: Counter() for s in ("train", "val", "test")}
    images, train_classes, bad = Counter(), {}, 0
    for i, (img, lbl, _) in enumerate(items):
        split = split_of[i]
        lines = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:        # polygons / malformed -> skip
                    bad += 1
                    continue
                lines.append(f"{remap[int(p[0])]} {' '.join(p[1:])}")
        if not lines:
            continue
        name = clean_stem(img.name) + img.suffix.lower()
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        dst = out / "images" / split / name
        shutil.copyfile(img, dst)      # real copy (symlinks would break on Windows)
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        images[split] += 1
        cls_ids = {int(l.split()[0]) for l in lines}
        for l in lines:
            boxes[split][CLASS_NAMES[int(l.split()[0])]] += 1
        if split == "train":
            train_classes[str(dst)] = cls_ids

    train_entry = "images/train"
    if a.oversample and train_classes:
        reps = repeat_factors(train_classes, a.oversample_t, a.seed)
        lines = [Path(p).resolve().as_posix() for p, k in sorted(reps.items()) for _ in range(k)]
        (out / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        train_entry = "train.txt"
        print(f"[oversample] train list: {len(train_classes)} unique -> {len(lines)} entries "
              "(NOTE: train.txt holds absolute paths of THIS machine; re-run on the training machine)")
    out_abs = out.resolve()
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(
        f"# Auto-generated by import_roboflow.py (source: {cfg.get('roboflow', {}).get('url', src.name)})\n"
        f"path: {out_abs.as_posix()}\ntrain: {train_entry}\nval: images/val\ntest: images/test\nnames:\n{names_block}\n",
        encoding="utf-8")
    stats = {"images": dict(images), "boxes": {s: dict(c) for s, c in boxes.items()},
             "split_by": a.split_by, "malformed_lines_skipped": bad, "oversample": a.oversample}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_distribution(stats, out / "class_distribution.png")
    if a.preview:
        write_previews(out, a.preview, a.seed)

    print("\n=== summary ===\nimages:", dict(images))
    print(f"{'class':<20}" + "".join(f"{s:>9}" for s in ("train", "val", "test")))
    for n in CLASS_NAMES:
        print(f"{n:<20}" + "".join(f"{boxes[s][n]:>9}" for s in ("train", "val", "test")))
    print(f"\ndata.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
