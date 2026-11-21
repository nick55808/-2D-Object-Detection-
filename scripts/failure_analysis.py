#!/usr/bin/env python3
"""Flag candidate images for manual failure analysis.

The ngdet harness stores per-image prediction/ground-truth counts. Counts do
not prove a semantic error, so this script only creates a prioritized audit
list. The final report must confirm candidates by inspecting the annotated
videos or images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("pred_*.json")):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in items:
            gt = int(item.get("num_gt", 0))
            pred = int(item.get("num_pred", 0))
            if gt == 0 and pred == 0:
                continue
            if gt > 0 and pred == 0:
                category = "no_predictions"
                priority = 3.0 + gt
            elif gt > 0 and pred < max(1, gt / 2):
                category = "prediction_count_below_gt"
                priority = 2.0 + (gt - pred)
            elif pred >= 2 * max(gt, 1):
                category = "many_predictions_relative_to_gt"
                priority = 1.0 + (pred - gt) / max(gt, 1)
            else:
                continue
            rows.append({
                "source_file": path.name,
                "image_id": item.get("image_id", ""),
                "num_gt": gt,
                "num_pred": pred,
                "candidate_type": category,
                "priority": round(priority, 3),
                "manual_verdict": "",
                "notes": "",
            })

    rows.sort(key=lambda row: float(row["priority"]), reverse=True)
    output = results_dir / "failure_candidates.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "source_file", "image_id", "num_gt", "num_pred", "candidate_type", "priority", "manual_verdict", "notes"
        ])
        writer.writeheader()
        writer.writerows(rows[:100])
    print(f"Wrote {min(100, len(rows))} manual failure-analysis candidates to {output}")


if __name__ == "__main__":
    main()
