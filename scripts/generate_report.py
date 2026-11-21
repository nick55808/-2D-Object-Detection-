#!/usr/bin/env python3
"""Render the submission report from a template and run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["No final results are available yet."] + ["N/A"] * (len(headers) - 1)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def environment_table(environment: dict[str, Any]) -> str:
    hardware = environment.get("hardware", {}) or {}
    hardware_value = "; ".join(
        f"{key}={value}" for key, value in hardware.items() if value
    ) or "N/A"
    rows = [
        ["Python", environment.get("python", "N/A").splitlines()[0]],
        ["Platform", environment.get("platform", "N/A")],
        ["GPU", "; ".join(environment.get("gpu", [])) or "No GPU recorded"],
        ["Hardware", hardware_value],
    ]
    accelerator = environment.get("accelerator", {})
    if accelerator:
        rows.append(["Accelerator", accelerator.get("selected_backend", "N/A")])
        rows.append(["CUDA available", accelerator.get("torch_cuda_available", "N/A")])
        rows.append(["CUDA runtime", accelerator.get("torch_cuda_runtime", "N/A") or "N/A"])
        rows.append(["CUDA device count", accelerator.get("cuda_device_count", "N/A")])
        devices = accelerator.get("cuda_devices", []) or []
        if devices:
            rows.append(["CUDA devices", "; ".join(
                f"{item.get('index')}: {item.get('name')} ({item.get('total_memory_gib')} GiB)"
                for item in devices
            )])
    for name, version in environment.get("packages", {}).items():
        rows.append([name, version or "not installed"])
    return table(["Item", "Recorded value"], rows)


def metric_table(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for item in metrics:
        meta = item.get("_meta", {})
        per_class = item.get("per_class", {}) or {}
        rows.append([
            meta.get("model", "unknown"),
            meta.get("dataset", "unknown"),
            meta.get("num_images", "N/A"),
            fmt(item.get("mAP")),
            fmt(item.get("AP50")),
            fmt(item.get("AP75")),
            fmt(item.get("AP_small")),
            fmt(item.get("AP_medium")),
            fmt(item.get("AP_large")),
            fmt(per_class.get("vehicle")),
            fmt(per_class.get("person")),
            fmt(per_class.get("cyclist")),
        ])
    return table(
        ["Model", "Dataset", "Images", "mAP", "AP50", "AP75", "AP small", "AP medium", "AP large", "Vehicle AP", "Person AP", "Cyclist AP"],
        rows,
    )


def latency_table(latency: list[dict[str, Any]]) -> str:
    rows = []
    for item in latency:
        rows.append([
            item.get("model", "unknown"),
            item.get("accel", "unknown"),
            fmt(item.get("mean_ms"), 1),
            fmt(item.get("p50_ms"), 1),
            fmt(item.get("p90_ms"), 1),
            fmt(item.get("fps"), 1),
        ])
    return table(["Model", "Mode", "Mean ms", "P50 ms", "P90 ms", "FPS"], rows)


def report_interpretation(metrics: list[dict[str, Any]], gaps: list[dict[str, str]], latency: list[dict[str, Any]]) -> str:
    if not metrics:
        return "Final interpretation is pending the GPU run. Do not submit the draft until the metrics and latency tables contain measured values."
    coco = [m for m in metrics if m.get("_meta", {}).get("dataset") == "coco"]
    kitti = [m for m in metrics if m.get("_meta", {}).get("dataset") == "kitti"]
    best_coco = max(coco, key=lambda m: m.get("mAP", -1), default=None)
    best_kitti = max(kitti, key=lambda m: m.get("mAP", -1), default=None)
    smallest = min(gaps, key=lambda row: float(row["domain_gap"]), default=None)
    fastest = max(latency, key=lambda row: row.get("fps", -1), default=None)
    paragraphs = []
    if best_coco:
        paragraphs.append(
            f"On COCO, {best_coco['_meta']['model']} achieved the highest measured mAP of {fmt(best_coco.get('mAP'))}."
        )
    if best_kitti:
        paragraphs.append(
            f"On KITTI, {best_kitti['_meta']['model']} achieved the highest measured mAP of {fmt(best_kitti.get('mAP'))}."
        )
    if smallest:
        paragraphs.append(
            f"The smallest measured COCO-to-KITTI gap was {fmt(smallest['domain_gap'])} for {smallest['model']}; this is the model with the strongest transfer under this evaluation protocol."
        )
    if fastest:
        paragraphs.append(
            f"The latency benchmark identified {fastest.get('model', 'N/A')} as the fastest measured configuration at {fmt(fastest.get('fps'), 1)} FPS."
        )
    return " ".join(paragraphs)


def conclusions(metrics: list[dict[str, Any]], gaps: list[dict[str, str]], latency: list[dict[str, Any]]) -> str:
    if not metrics:
        return "The experiment code and report structure are complete, but measured conclusions cannot be written until the final GPU run produces metrics.json and latency.json."
    sentences = [
        "The experiment provides a controlled zero-shot comparison because all models use the same mapped taxonomy, dataset samples, score threshold, and evaluation metrics.",
    ]
    if gaps:
        best_transfer = min(gaps, key=lambda row: float(row["domain_gap"]))
        sentences.append(
            f"The strongest transfer model under the measured protocol was {best_transfer['model']}, with a COCO-to-KITTI mAP gap of {fmt(best_transfer['domain_gap'])}."
        )
    if latency:
        fastest = max(latency, key=lambda row: row.get("fps", -1))
        sentences.append(
            f"The fastest configuration was {fastest.get('model', 'N/A')} at {fmt(fastest.get('fps'), 1)} FPS, but speed should be interpreted together with its accuracy and failure profile."
        )
    sentences.append(
        "The main limitations are zero-shot domain shift, the coarse class mapping, a fixed confidence threshold, and the use of a labeled KITTI split rather than a hidden test server."
    )
    return " ".join(sentences)


def figures(results_dir: Path) -> str:
    candidates = [
        ("heatmap_mAP.png", "Model-by-dataset mAP heatmap"),
        ("domain_gap.png", "Cross-dataset generalization gap"),
        ("speed_accuracy.png", "Accuracy-latency tradeoff"),
    ]
    found = [f"![{caption}]({name})" for name, caption in candidates if (results_dir / name).exists()]
    return "\n\n".join(found) or "Figures will appear here after the final analysis script runs."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    results_dir = Path(args.results_dir)
    metrics = load_json(results_dir / "metrics.json", [])
    latency = load_json(results_dir / "latency.json", [])
    environment = load_json(Path(args.environment), {})
    config = load_json(Path(args.config), {})
    gaps = read_csv(results_dir / "domain_gap.csv")
    failures = read_csv(results_dir / "failure_candidates.csv")

    status = (
        "Measured final results are present."
        if metrics
        else "DRAFT STATUS: no final metrics.json is present; run the experiment before submitting."
    )
    failure_text = (
        f"The automated audit identified {len(failures)} candidate images. These are only triage signals based on prediction/ground-truth counts. Each candidate must be visually confirmed in the annotated video, and the final report should record at least five confirmed cases with an explanation."
        if failures
        else "The failure-audit table will be populated after the final run. The intended manual audit categories are missed objects, duplicate detections, localization errors, taxonomy mismatches, small-object misses, and false positives."
    )

    replacements = {
        "{{TITLE}}": config.get("title", "Zero-Shot Cross-Dataset Evaluation of Object Detectors"),
        "{{STUDENT}}": config.get("student_name", "Student"),
        "{{COURSE}}": config.get("course", "2D Object Detection Homework"),
        "{{STATUS}}": status,
        "{{ENVIRONMENT_TABLE}}": environment_table(environment),
        "{{RESULTS_TABLE}}": metric_table(metrics),
        "{{LATENCY_TABLE}}": latency_table(latency),
        "{{DOMAIN_GAP_TABLE}}": table(
            ["Model", "COCO mAP", "KITTI mAP", "Gap"],
            [[row["model"], fmt(row.get("coco_mAP")), fmt(row.get("kitti_mAP")), fmt(row.get("domain_gap"))] for row in gaps],
        ),
        "{{INTERPRETATION}}": report_interpretation(metrics, gaps, latency),
        "{{FAILURE_TEXT}}": failure_text,
        "{{FIGURES}}": figures(results_dir),
        "{{CONCLUSIONS}}": conclusions(metrics, gaps, latency),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
