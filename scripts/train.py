#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train queen bee detector.")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/queen-bee-detector")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = REPO_ROOT / project_path

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
