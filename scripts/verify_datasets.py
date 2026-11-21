#!/usr/bin/env python3
"""Validate tutorial and dataset paths before an evaluation can start."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str, base: Path = ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()


def check(condition: bool, message: str, errors: list[str]) -> None:
    prefix = "OK" if condition else "ERROR"
    print(f"[{prefix}] {message}")
    if not condition:
        errors.append(message)


def first_file(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        match = next(directory.glob(pattern), None)
        if match is not None:
            return match
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    errors: list[str] = []
    check(config_path.is_file(), f"config file exists: {config_path}", errors)
    if errors:
        return 1

    cfg: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    tutorial_root = resolve_path(cfg["tutorial_root"])
    check(
        (tutorial_root / "DeepDataMiningLearning").is_dir(),
        f"tutorial package exists: {tutorial_root / 'DeepDataMiningLearning'}",
        errors,
    )

    roots = cfg.get("dataset_roots", {})
    coco_root = resolve_path(roots.get("coco", ""))
    coco_ann = resolve_path(cfg.get("coco_ann", ""))
    check(coco_root.is_dir(), f"COCO image directory exists: {coco_root}", errors)
    check(coco_ann.is_file(), f"COCO annotation file exists: {coco_ann}", errors)
    if coco_root.is_dir():
        image = first_file(coco_root, ("*.jpg", "*.jpeg", "*.png"))
        check(image is not None, f"COCO image directory contains images: {coco_root}", errors)
    if coco_ann.is_file():
        try:
            annotations = json.loads(coco_ann.read_text(encoding="utf-8"))
            check(bool(annotations.get("images")), "COCO annotations contain images", errors)
            check(bool(annotations.get("annotations")), "COCO annotations contain boxes", errors)
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"COCO annotation JSON is readable: {exc}")
            print(f"[ERROR] COCO annotation JSON is readable: {exc}")

    kitti_root = resolve_path(roots.get("kitti", ""))
    kitti_images = kitti_root / "training" / "image_2"
    kitti_labels = kitti_root / "training" / "label_2"
    check(kitti_images.is_dir(), f"KITTI image directory exists: {kitti_images}", errors)
    check(kitti_labels.is_dir(), f"KITTI label directory exists: {kitti_labels}", errors)
    if kitti_images.is_dir():
        check(any(kitti_images.glob("*.png")), f"KITTI image directory contains PNG files: {kitti_images}", errors)
    if kitti_labels.is_dir():
        check(any(kitti_labels.glob("*.txt")), f"KITTI label directory contains TXT files: {kitti_labels}", errors)

    latency_image = resolve_path(cfg.get("latency_image", ""))
    check(latency_image.is_file(), f"latency image exists: {latency_image}", errors)

    if errors:
        print("\nDataset preflight failed. Correct config.json and rerun this command.")
        return 1
    print("\nDataset preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

