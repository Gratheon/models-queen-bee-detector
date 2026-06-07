#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from roboflow import Roboflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Roboflow object detection dataset.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--output", default="datasets/queen-bee")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is required. Create a Roboflow API key and export it first.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=str(output))

    candidates = [Path(dataset.location) / "data.yaml", output / "data.yaml"]
    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        raise SystemExit(f"Dataset downloaded to {dataset.location}, but data.yaml was not found")

    print(yaml_path)


if __name__ == "__main__":
    main()
