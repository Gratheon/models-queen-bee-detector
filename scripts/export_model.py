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
    parser = argparse.ArgumentParser(description="Export queen bee detector.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "tfjs", "tflite", "torchscript", "openvino"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    weights_path = resolve_repo_path(args.weights)
    model = YOLO(str(weights_path))
    exported = model.export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)
    print(exported)


if __name__ == "__main__":
    main()
