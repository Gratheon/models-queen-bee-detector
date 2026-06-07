#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate queen bee detector.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--project", default="runs/queen-bee-detector")
    parser.add_argument("--name", default="eval")
    args = parser.parse_args()

    weights_path = resolve_repo_path(args.weights)
    data_path = resolve_repo_path(args.data)
    project_path = resolve_repo_path(args.project)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        split=args.split,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
    )

    print(
        {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "save_dir": str(metrics.save_dir),
        }
    )


if __name__ == "__main__":
    main()
