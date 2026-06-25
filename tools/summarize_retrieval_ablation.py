from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize retrieval-repair ablations from saved inference runs.")
    p.add_argument("--run", action="append", required=True, help="name=run_dir")
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Expected name=run_dir, got {item}")
        name, value = item.split("=", 1)
        run_dir = Path(value)
        metrics = read_json(run_dir / "metrics.json")
        roi_rows = read_json(run_dir / "roi_budget.json")
        repaired = [row for row in roi_rows if float(row.get("action_latency_ms", 0.0)) > 0.0]
        rows.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "image_auroc": metrics.get("image_auroc"),
                "pixel_auroc": metrics.get("pixel_auroc"),
                "pixel_ap": metrics.get("pixel_ap"),
                "aupro": metrics.get("aupro"),
                "dice": metrics.get("dice"),
                "latency_ms": metrics.get("eff_latency_ms_mean"),
                "nfe": metrics.get("eff_nfe_mean"),
                "retrieval_enabled": metrics.get("retrieval_repair_enabled"),
                "retrieval_mode": metrics.get("retrieval_reference_mode"),
                "retrieval_similarity": metrics.get("retrieval_similarity_mean"),
                "retrieval_weight": metrics.get("retrieval_effective_weight_mean"),
                "repair_count": len(repaired),
                "repair_gain_mean": (
                    sum(float(row.get("repair_gain", 0.0)) for row in repaired) / len(repaired)
                    if repaired
                    else 0.0
                ),
            }
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (out / "retrieval_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "runs": rows,
        "paper_status": "negative_ablation_not_enabled_by_default",
        "uses_real_anomaly_labels_for_configuration": False,
    }
    (out / "retrieval_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
