#!/usr/bin/env python3
"""Fail-fast CUDA and RTX verification for the Track A experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch is not installed in this interpreter: {exc}", file=sys.stderr)
        return 2

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            print("nvidia-smi:")
            print(smi.stdout.strip())
        else:
            print("nvidia-smi: unavailable or returned an error")
    except OSError:
        print("nvidia-smi: command not found")

    if not torch.cuda.is_available():
        print(
            "CUDA verification failed. Confirm that the NVIDIA driver is installed, "
            "the active interpreter is the project .venv, and a CUDA-enabled "
            "PyTorch wheel was installed.",
            file=sys.stderr,
        )
        return 1

    try:
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = 0
        if device_index >= torch.cuda.device_count():
            raise IndexError(f"device index {device_index} is not available")
        props = torch.cuda.get_device_properties(device_index)
        print(f"GPU: {props.name}")
        print(f"GPU index: {device_index}")
        print(f"VRAM: {props.total_memory / (1024 ** 3):.2f} GiB")
        print(f"Compute capability: {props.major}.{props.minor}")

        with torch.cuda.device(device_index):
            x = torch.randn((512, 512), device=f"cuda:{device_index}")
            y = x @ x
            torch.cuda.synchronize(device_index)
        print(f"GPU test passed; tensor device: {y.device}")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic command
        print(f"CUDA GPU test failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

