#!/usr/bin/env python3
"""Collect hardware and Python-package information for reproducibility."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_accelerator_info() -> dict[str, object]:
    """Collect CUDA device information for reproducibility."""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices: list[dict[str, object]] = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        return {
            "torch_cuda_available": cuda_available,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_devices": devices,
            "selected_backend": "cuda" if cuda_available else "cpu",
        }
    except Exception as exc:  # noqa: BLE001 - environment capture must not abort
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nvidia = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    data = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "gpu": nvidia.splitlines() if nvidia else [],
        "hardware": {
            "machine": platform.machine(),
            "release": platform.release(),
            "processor": platform.processor(),
        },
        "accelerator": torch_accelerator_info(),
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "torchvision",
                "transformers",
                "ultralytics",
                "pycocotools",
                "opencv-python",
                "matplotlib",
                "numpy",
                "pandas",
                "pyarrow",
                "Pillow",
            ]
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
