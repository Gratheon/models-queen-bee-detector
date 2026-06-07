#!/usr/bin/env python3
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    queen_class_ids: set[int]
    prefix: str


def read_names(root: Path) -> list[str]:
    data = yaml.safe_load((root / "data.yaml").read_text())
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def queen_class_ids(root: Path) -> set[int]:
    ids = set()
    for idx, name in enumerate(read_names(root)):
        normalized = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
        if "queen" in normalized:
            ids.add(idx)
    if not ids:
        raise ValueError(f"No queen-like class found in {root / 'data.yaml'}")
    return ids


def image_files(path: Path) -> Iterable[Path]:
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        yield from path.glob(pattern)


def convert_to_bbox(parts: list[str]) -> list[str] | None:
    """Return YOLO bbox coordinates from bbox or polygon label parts."""
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        return [str(value) for value in coords]
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return None
        return [str(x_center), str(y_center), str(width), str(height)]
    return None


def convert_label(src_label: Path, dst_label: Path, queen_ids: set[int]) -> int:
    converted = []
    if src_label.exists():
        for line in src_label.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            if class_id in queen_ids:
                bbox = convert_to_bbox(parts)
                if bbox:
                    converted.append("0 " + " ".join(bbox))
    dst_label.write_text("\n".join(converted) + ("\n" if converted else ""))
    return len(converted)


def merge_split(sources: list[SourceDataset], split: str, output_root: Path) -> tuple[int, int]:
    dst_images = output_root / split / "images"
    dst_labels = output_root / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    queen_count = 0
    for source in sources:
        src_images = source.root / split / "images"
        src_labels = source.root / split / "labels"
        if not src_images.exists():
            continue

        for image in image_files(src_images):
            stem = f"{source.prefix}_{image.stem}"
            dst_image = dst_images / f"{stem}{image.suffix.lower()}"
            dst_label = dst_labels / f"{stem}.txt"
            shutil.copy2(image, dst_image)
            queen_count += convert_label(src_labels / f"{image.stem}.txt", dst_label, source.queen_class_ids)
            image_count += 1
    return image_count, queen_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into a single-class queen detector dataset.")
    parser.add_argument("--output", default="datasets/queen-bee-merged")
    parser.add_argument("sources", nargs="+", help="Source dataset roots containing data.yaml")
    args = parser.parse_args()

    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    sources = [
        SourceDataset(root=Path(source), queen_class_ids=queen_class_ids(Path(source)), prefix=f"src{idx}")
        for idx, source in enumerate(args.sources)
    ]

    summary = {}
    for split in ["train", "valid", "test"]:
        images, queens = merge_split(sources, split, output_root)
        summary[split] = {"images": images, "queen_objects": queens}

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["queen"],
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (output_root / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    print(output_root / "data.yaml")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
