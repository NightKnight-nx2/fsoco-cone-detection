"""Evaluate a trained model: mAP50 / mAP50-95, per-class AP, confusion matrix, sample predictions.

Outputs in ``--out`` (default ``results/``)::

    metrics.json, metrics.md          overall + per-class precision / recall / AP50 / AP50-95, speed
    confusion_matrix*.png, PR_curve.png, ...   Ultralytics validation plots
    predictions/gt_vs_pred_XX.jpg     ground truth (left) | prediction (right), drawn with our clean style
    predictions_grid.jpg              2x2 prediction montage for the README

Example::

    python evaluate.py --weights runs/fsoco_yolo11n/weights/best.pt --data data/fsoco_yolo/data.yaml --split test
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import (CLASS_NAMES, draw_detections, imread, imwrite, label_path_for,
                    read_yolo_labels, resolve_split_images, select_device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for mAP (standard practice)")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--samples", type=int, default=6, help="number of GT-vs-prediction images")
    ap.add_argument("--pred-conf", type=float, default=0.25, help="confidence threshold for drawn predictions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(a.device)
    model = YOLO(a.weights)

    m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, conf=a.conf, iou=a.iou,
                  device=device, plots=True, project=str(out), name="val_plots", exist_ok=True,
                  verbose=False)
    box = m.box
    per_class = {}
    for k, ci in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[int(ci)]] = {
            "precision": float(box.p[k]), "recall": float(box.r[k]),
            "ap50": float(box.ap50[k]), "ap50_95": float(box.ap[k]),
        }
    metrics = {
        "weights": str(a.weights), "split": a.split, "imgsz": a.imgsz,
        "precision": float(box.mp), "recall": float(box.mr),
        "mAP50": float(box.map50), "mAP50-95": float(box.map),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in m.speed.items()},
        "device": str(device),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    md = [f"| Class | Precision | Recall | AP50 | AP50-95 |", "|---|---:|---:|---:|---:|"]
    for n in CLASS_NAMES:
        if n in per_class:
            v = per_class[n]
            md.append(f"| {n} | {v['precision']:.3f} | {v['recall']:.3f} | {v['ap50']:.3f} | {v['ap50_95']:.3f} |")
    md.append(f"| **all** | **{metrics['precision']:.3f}** | **{metrics['recall']:.3f}** | "
              f"**{metrics['mAP50']:.3f}** | **{metrics['mAP50-95']:.3f}** |")
    (out / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    save_dir = Path(m.save_dir)
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"):
        if (save_dir / f).exists():
            shutil.copy(save_dir / f, out / f)

    # ---- qualitative samples: ground truth | prediction ----------------------------------- #
    imgs = resolve_split_images(a.data, a.split)
    random.Random(a.seed).shuffle(imgs)
    imgs = imgs[: a.samples]
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    preds = []
    for i, p in enumerate(imgs):
        img = imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=a.imgsz, conf=a.pred_conf, iou=a.iou, device=device, verbose=False)[0]
        gt_xyxy, gt_cls = read_yolo_labels(label_path_for(p), w, h)
        left = draw_detections(img.copy(), gt_xyxy, gt_cls)
        right = draw_detections(img.copy(), r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy())
        for im, t in ((left, "ground truth"), (right, "prediction")):
            cv2.putText(im, t, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1600), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(im, t, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1600), (0, 0, 0), 1, cv2.LINE_AA)
        imwrite(pred_dir / f"gt_vs_pred_{i:02d}.jpg", np.hstack([left, right]))
        preds.append(right)
    if len(preds) >= 4:
        tile = [cv2.resize(p, (640, 360)) for p in preds[:4]]
        imwrite(out / "predictions_grid.jpg", np.vstack([np.hstack(tile[:2]), np.hstack(tile[2:])]))

    print("\n" + (out / "metrics.md").read_text())
    print(f"mAP50 = {metrics['mAP50']:.4f}   mAP50-95 = {metrics['mAP50-95']:.4f}   "
          f"inference = {metrics['speed_ms_per_image'].get('inference', float('nan'))} ms/img on {device}")
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
