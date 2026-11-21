#!/usr/bin/env python3
"""Create tables and plots from ngdet metrics and latency outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRICS = [
    "mAP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def model_family(label: str) -> str:
    value = label.lower()
    if "fasterrcnn" in value or "rcnn" in value:
        return "Two-stage CNN"
    if "yolo" in value:
        return "One-stage CNN"
    if "rtdetr" in value or "detr" in value:
        return "Transformer"
    return "Other"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results_dir: Path, metrics: list[dict[str, Any]], gaps: list[dict[str, Any]], latency: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return

    if metrics:
        models = []
        datasets = []
        values: dict[tuple[str, str], float] = {}
        for row in metrics:
            model = row["model"]
            dataset = row["dataset"]
            models.append(model) if model not in models else None
            datasets.append(dataset) if dataset not in datasets else None
            values[(model, dataset)] = float(row["mAP"])
        grid = [[values.get((model, ds), float("nan")) for ds in datasets] for model in models]
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(datasets)), max(3.5, 0.65 * len(models) + 1.5)))
        image = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(datasets)), datasets)
        ax.set_yticks(range(len(models)), models)
        ax.set_title("Zero-shot mAP by model and dataset")
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                if math.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.5 else "black")
        fig.colorbar(image, ax=ax, label="mAP@[0.50:0.95]")
        fig.tight_layout()
        fig.savefig(results_dir / "heatmap_mAP.png", dpi=180)
        plt.close(fig)

    if gaps:
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(gaps)), 4.5))
        labels = [row["model"] for row in gaps]
        values = [float(row["domain_gap"]) for row in gaps]
        bars = ax.bar(labels, values, color="#4f46e5")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("COCO mAP - KITTI mAP")
        ax.set_title("Cross-dataset generalization gap")
        ax.tick_params(axis="x", rotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(results_dir / "domain_gap.png", dpi=180)
        plt.close(fig)

    if latency and metrics:
        accuracy = {(row["model"], row["dataset"]): row for row in metrics}
        coco = {row["model"]: row for row in metrics if row["dataset"] == "coco"}
        pairs = []
        for row in latency:
            if row["model"] in coco:
                pairs.append((float(coco[row["model"]]["mAP"]), float(row["mean_ms"]), row["model"]))
        if pairs:
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            for ap, ms, label in pairs:
                ax.scatter(ms, ap, s=80)
                ax.annotate(label, (ms, ap), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.set_xlabel("End-to-end latency (ms/image)")
            ax.set_ylabel("COCO mAP@[0.50:0.95]")
            ax.set_title("Accuracy-latency tradeoff")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(results_dir / "speed_accuracy.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_raw = load_json(results_dir / "metrics.json", [])
    latency_raw = load_json(results_dir / "latency.json", [])

    metrics: list[dict[str, Any]] = []
    for item in metrics_raw:
        meta = item.get("_meta", {})
        model = meta.get("model", "unknown")
        row: dict[str, Any] = {
            "model": model,
            "family": model_family(model),
            "dataset": meta.get("dataset", "unknown"),
            "num_images": meta.get("num_images", ""),
        }
        for key in METRICS:
            row[key] = item.get(key)
        for name, value in (item.get("per_class") or {}).items():
            row[f"AP_{name}"] = value
        metrics.append(row)

    fieldnames = ["model", "family", "dataset", "num_images", *METRICS]
    class_names = sorted({key for row in metrics for key in row if key.startswith("AP_") and key not in METRICS})
    fieldnames.extend(class_names)
    write_csv(results_dir / "summary.csv", metrics, fieldnames)

    by_pair = {(row["model"], row["dataset"]): row for row in metrics}
    models = sorted({row["model"] for row in metrics})
    gaps: list[dict[str, Any]] = []
    for model in models:
        coco = by_pair.get((model, "coco"))
        kitti = by_pair.get((model, "kitti"))
        if coco and kitti and coco.get("mAP") is not None and kitti.get("mAP") is not None:
            gaps.append({
                "model": model,
                "coco_mAP": float(coco["mAP"]),
                "kitti_mAP": float(kitti["mAP"]),
                "domain_gap": float(coco["mAP"]) - float(kitti["mAP"]),
            })
    write_csv(results_dir / "domain_gap.csv", gaps, ["model", "coco_mAP", "kitti_mAP", "domain_gap"])

    latency = []
    for row in latency_raw:
        latency.append({
            "model": row.get("model", "unknown"),
            "accel": row.get("accel", "unknown"),
            "mean_ms": row.get("mean_ms"),
            "p50_ms": row.get("p50_ms"),
            "p90_ms": row.get("p90_ms"),
            "fps": row.get("fps"),
        })

    analysis = {
        "num_metric_rows": len(metrics),
        "num_latency_rows": len(latency),
        "best_coco_model": max((row for row in metrics if row["dataset"] == "coco"), key=lambda r: r.get("mAP") or -1, default=None),
        "best_kitti_model": max((row for row in metrics if row["dataset"] == "kitti"), key=lambda r: r.get("mAP") or -1, default=None),
        "smallest_domain_gap": min(gaps, key=lambda r: r["domain_gap"], default=None),
        "largest_domain_gap": max(gaps, key=lambda r: r["domain_gap"], default=None),
        "fastest_model": max(latency, key=lambda r: r.get("fps") or -1, default=None),
    }
    (results_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    plot_results(results_dir, metrics, gaps, latency)
    print(f"Analyzed {len(metrics)} accuracy rows and {len(latency)} latency rows in {results_dir}")


if __name__ == "__main__":
    main()
