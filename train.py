"""Fine-tune a nano YOLO model (YOLO11n / YOLOv8n) on the converted FSOCO dataset.

Design choices (all overridable from the CLI):

* **640x640 input, nano model** - small enough for real-time inference on an embedded
  GPU, which is what a Formula Student driverless car actually runs.
* **COCO-pretrained weights** are fine-tuned (transfer learning converges far faster
  than training from scratch).
* **No hue augmentation** (``hsv_h=0``). Hue shifts would turn blue cones yellow and
  destroy the label semantics; saturation/value jitter is kept for lighting robustness.
* **Class imbalance**: Ultralytics has no per-class loss weights, so imbalance is handled
  in the data (``data_prep.py --oversample`` = repeat-factor sampling) plus mosaic / scale
  / translate augmentation. ``--cls`` raises the classification-loss gain if rare classes
  (large_orange / unknown) still lag.
* SGD with ``lr0=0.01`` and cosine decay. (Ultralytics' ``optimizer=auto`` silently ignores
  ``lr0``, so the optimiser is set explicitly.)

Examples::

    python train.py --data data/fsoco_yolo/data.yaml                      # full run, 100 epochs
    python train.py --data data/fsoco_small/data.yaml --epochs 3 --batch 8 --name smoke
    python train.py --data ... --model yolov8n.pt --batch -1               # YOLOv8n, auto batch
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml written by data_prep.py")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt (default) or yolov8n.pt, or a .yaml for scratch")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto-fit to GPU memory")
    ap.add_argument("--device", default="auto", help="auto | 0 | 0,1 | cpu | mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default="SGD", help="SGD | AdamW | auto (auto ignores --lr0!)")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01, help="final lr = lr0 * lrf")
    ap.add_argument("--cos-lr", action="store_true", default=True)
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    ap.add_argument("--cls", type=float, default=0.5, help="classification loss gain (raise for rare classes)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for the last N epochs")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set to use")
    ap.add_argument("--cache", default="False", help="False | ram | disk")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="fsoco_yolo11n")
    ap.add_argument("--resume", action="store_true", help="resume the last interrupted run of --name")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO

    device = select_device(a.device)
    if device == "cpu":
        print("WARNING: training on CPU is very slow. Use a CUDA GPU, or the Colab notebook "
              "(notebooks/colab_train.ipynb). CPU is fine for the --epochs 1-3 smoke test.")
    cache = {"False": False, "false": False, "ram": "ram", "disk": "disk", "True": "ram"}.get(a.cache, False)

    # Ultralytics prefixes a *relative* project with its task folder (runs/detect/<project>/...);
    # an absolute path keeps the layout predictable: <project>/<name>/weights/best.pt
    project = str(Path(a.project).resolve())

    if a.resume:
        last = Path(project) / a.name / "weights" / "last.pt"
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=device,
        workers=a.workers, optimizer=a.optimizer, lr0=a.lr0, lrf=a.lrf, cos_lr=a.cos_lr,
        momentum=0.937, weight_decay=5e-4, warmup_epochs=3, patience=a.patience, cls=a.cls,
        # --- augmentation strategy for cones ---
        hsv_h=0.0,            # never shift hue: blue <-> yellow would corrupt labels
        hsv_s=0.6, hsv_v=0.4, # lighting / saturation robustness (sun, shade, wet asphalt)
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.0, close_mosaic=a.close_mosaic, erasing=0.2,
        # --- bookkeeping ---
        fraction=a.fraction, cache=cache, project=project, name=a.name, exist_ok=True,
        seed=a.seed, deterministic=True, plots=True, amp=True, pretrained=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nTraining finished. Best weights: {best}")
    print(f"Next: python evaluate.py --weights {best} --data {a.data}")


if __name__ == "__main__":  # required on Windows (DataLoader worker processes)
    main()
