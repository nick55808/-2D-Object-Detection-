#!/usr/bin/env python3
"""Run and package the Track A object-detection experiment.

This script delegates model evaluation to the tutorial repository's ngdet
implementation and adds reproducibility bookkeeping, latency measurement,
result analysis, failure-candidate extraction, and report generation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    rendered = shlex.join(command)
    stamp = datetime.now(timezone.utc).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] cwd={cwd}\n$ {rendered}\n")
        print(f"\n$ {rendered}\n")
        if dry_run:
            return
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
        log.write(f"[returncode={code}]\n")
        if code != 0:
            raise RuntimeError(f"Command failed with return code {code}: {rendered}")


def build_eval_command(cfg: dict[str, Any], phase_dir: Path, max_images: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "DeepDataMiningLearning.ngdet.run_eval",
        "--models",
        *cfg["models"],
        "--datasets",
        *cfg["datasets"],
        "--taxonomy",
        cfg.get("taxonomy", "driving3"),
        "--max-images",
        str(max_images),
        "--score-thr",
        str(cfg.get("score_thr", 0.3)),
        "--device",
        cfg.get("device", "cuda"),
        "--out-dir",
        str(phase_dir),
    ]
    if cfg.get("video", False):
        command.append("--video")
        command += ["--video-max", str(cfg.get("video_max", 40))]
    if cfg.get("coco_ann"):
        command += ["--coco-ann", str(resolve_path(cfg["coco_ann"], ROOT))]
    roots = cfg.get("dataset_roots", {})
    if roots:
        command.append("--roots")
        for name in cfg["datasets"]:
            if name in roots:
                command.append(f"{name}={resolve_path(roots[name], ROOT)}")
    return command


def build_latency_command(cfg: dict[str, Any], phase_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "DeepDataMiningLearning.ngdet.latency",
        "--models",
        *cfg["models"],
        "--accel",
        "fp32",
        "--image",
        str(resolve_path(cfg["latency_image"], ROOT)),
        "--warmup",
        str(cfg.get("latency_warmup", 10)),
        "--iters",
        str(cfg.get("latency_iters", 50)),
        "--device",
        cfg.get("device", "cuda"),
        "--taxonomy",
        cfg.get("taxonomy", "driving3"),
        "--out-dir",
        str(phase_dir),
    ]


def run_python_script(script: Path, args: list[str], log_path: Path, dry_run: bool) -> None:
    run_command([sys.executable, str(script), *args], ROOT, log_path, dry_run)


def validate_requested_device(cfg: dict[str, Any]) -> None:
    """Fail before model downloads when config requests unavailable CUDA."""
    device = str(cfg.get("device", "cuda"))
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The config requests CUDA, but PyTorch is not installed in this interpreter. "
            "Install the CUDA-enabled torch wheel inside the active .venv."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The config requests CUDA, but torch.cuda.is_available() is False. "
            "Run `nvidia-smi` and `python scripts/verify_cuda.py`, then reinstall "
            "the matched CUDA-enabled PyTorch build if necessary."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--phase", choices=["smoke", "final", "all"], default="final")
    parser.add_argument("--tutorial-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config, ROOT)
    cfg = read_json(cfg_path)
    if not args.dry_run:
        validate_requested_device(cfg)
    tutorial_root = resolve_path(args.tutorial_root or cfg["tutorial_root"], ROOT)
    if not args.dry_run and not (tutorial_root / "DeepDataMiningLearning").exists():
        raise FileNotFoundError(
            f"Tutorial repository not found at {tutorial_root}. Clone it or pass --tutorial-root."
        )

    out_root = resolve_path(cfg.get("out_dir", "results"), ROOT)
    log_path = out_root / "commands.log"
    out_root.mkdir(parents=True, exist_ok=True)

    environment_script = ROOT / "scripts" / "collect_environment.py"
    run_python_script(
        ROOT / "scripts" / "verify_datasets.py",
        ["--config", str(cfg_path)],
        log_path,
        args.dry_run,
    )
    run_python_script(
        environment_script,
        ["--out", str(out_root / "environment.json")],
        log_path,
        args.dry_run,
    )

    phases = ["smoke", "final"] if args.phase == "all" else [args.phase]
    for phase in phases:
        phase_dir = out_root / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        max_images = (
            cfg.get("max_images_smoke", 50)
            if phase == "smoke"
            else cfg.get("max_images_final", 500)
        )
        run_command(
            build_eval_command(cfg, phase_dir, max_images),
            tutorial_root,
            log_path,
            args.dry_run,
        )

        if phase == "final":
            run_command(
                build_latency_command(cfg, phase_dir),
                tutorial_root,
                log_path,
                args.dry_run,
            )

        metrics = phase_dir / "metrics.json"
        if args.dry_run or metrics.exists():
            run_python_script(
                ROOT / "scripts" / "analyze_results.py",
                ["--results-dir", str(phase_dir)],
                log_path,
                args.dry_run,
            )
            run_python_script(
                ROOT / "scripts" / "failure_analysis.py",
                ["--results-dir", str(phase_dir)],
                log_path,
                args.dry_run,
            )

    final_dir = out_root / "final"
    report_path = final_dir / "final_report.md"
    run_python_script(
        ROOT / "scripts" / "generate_report.py",
        [
            "--template",
            str(ROOT / "report" / "technical_report_template.md"),
            "--results-dir",
            str(final_dir),
            "--environment",
            str(out_root / "environment.json"),
            "--config",
            str(cfg_path),
            "--out",
            str(report_path),
        ],
        log_path,
        args.dry_run,
    )
    print(f"\nReport Markdown: {report_path}")
    print("Generate the PDF with make_pdf.py after the final run completes.")


if __name__ == "__main__":
    main()
