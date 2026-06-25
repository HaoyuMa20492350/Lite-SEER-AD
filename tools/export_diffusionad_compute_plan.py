from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the full DiffusionAD MVTec15 compute and storage plan "
            "from a measured resumable training segment."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="SEER-AD-dataset/MVTec-AD",
    )
    parser.add_argument(
        "--timing-history",
        default=(
            "baselines/external_outputs/mvtec15/diffusionad/"
            "toothbrush/training_history.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "baselines/external_outputs/mvtec15/diffusionad/"
            "toothbrush/training_checkpoint.pth"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--out",
        default="tables/diffusionad_compute_plan",
    )
    return parser.parse_args()


def measured_seconds_per_batch(history: dict[str, Any]) -> float:
    start_epoch = int(history["resumed_from_epoch"])
    end_epoch = int(history["epochs"])
    completed_epochs = end_epoch - start_epoch
    if completed_epochs <= 0:
        raise ValueError("Timing history contains no completed timing epochs")
    records = [
        row
        for row in history["history"]
        if start_epoch < int(row["epoch"]) <= end_epoch
    ]
    if len(records) != completed_epochs:
        raise ValueError(
            "Timing history does not contain every measured epoch: "
            f"{len(records)} vs {completed_epochs}"
        )
    batches = sum(int(row["batches"]) for row in records)
    if batches <= 0:
        raise ValueError("Timing history contains no effective batches")
    return float(history["training_seconds_this_invocation"]) / batches


def category_plan(
    category: str,
    train_images: int,
    *,
    batch_size: int,
    epochs: int,
    seconds_per_batch: float,
) -> dict[str, Any]:
    batches_per_epoch = train_images // batch_size
    if batches_per_epoch < 1:
        raise ValueError(f"{category} has no full training batch")
    steps = batches_per_epoch * epochs
    seconds = steps * seconds_per_batch
    return {
        "category": category,
        "train_images": train_images,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "optimizer_steps": steps,
        "estimated_gpu_hours": seconds / 3600.0,
        "estimated_gpu_days": seconds / 86400.0,
    }


def greedy_walltime(
    category_rows: list[dict[str, Any]],
    gpu_count: int,
) -> float:
    loads = [0.0] * gpu_count
    for row in sorted(
        category_rows,
        key=lambda item: float(item["estimated_gpu_hours"]),
        reverse=True,
    ):
        index = min(range(gpu_count), key=loads.__getitem__)
        loads[index] += float(row["estimated_gpu_hours"])
    return max(loads)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    timing_path = Path(args.timing_history)
    checkpoint_path = Path(args.checkpoint)
    history = json.loads(timing_path.read_text(encoding="utf-8"))
    seconds_per_batch = measured_seconds_per_batch(history)
    rows = []
    for category_dir in sorted(
        path for path in dataset_root.iterdir() if path.is_dir()
    ):
        train_images = len(
            list((category_dir / "train" / "good").glob("*.png"))
        )
        rows.append(
            category_plan(
                category_dir.name,
                train_images,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seconds_per_batch=seconds_per_batch,
            )
        )
    total_gpu_hours = sum(
        float(row["estimated_gpu_hours"]) for row in rows
    )
    checkpoint_bytes = checkpoint_path.stat().st_size
    scenarios = []
    for gpu_count in (1, 4, 8, 15):
        wall_hours = greedy_walltime(rows, gpu_count)
        scenarios.append(
            {
                "gpu_count": gpu_count,
                "ideal_greedy_wall_hours": wall_hours,
                "ideal_greedy_wall_days": wall_hours / 24.0,
                "aggregate_gpu_hours": total_gpu_hours,
            }
        )
    summary = {
        "protocol": "diffusionad_measured_compute_plan_v1",
        "timing_history": str(timing_path),
        "timing_resumed_from_epoch": history["resumed_from_epoch"],
        "timing_end_epoch": history["epochs"],
        "seconds_per_effective_batch": seconds_per_batch,
        "author_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "epochs_per_category": args.epochs,
        "categories": len(rows),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in rows),
        "estimated_total_gpu_hours": total_gpu_hours,
        "estimated_total_gpu_days": total_gpu_hours / 24.0,
        "checkpoint_bytes_per_category": checkpoint_bytes,
        "estimated_checkpoint_storage_gib": (
            checkpoint_bytes * len(rows) / 1024**3
        ),
        "hardware_fit": {
            "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU 16GB",
            "full_batch_16_fp32": "out_of_memory",
            "full_batch_16_amp": "out_of_memory",
            "full_batch_16_amp_activation_checkpointing": "out_of_memory",
            "full_batch_16_amp_saved_tensor_offload": "out_of_memory",
            "micro_batch_8_amp": "out_of_memory",
            "effective_batch_16_micro_batch_4_amp": "stable",
            "optimizer_steps_per_author_batch": 1,
            "batchnorm_statistics": "micro_batch_4",
        },
        "parallel_scenarios": scenarios,
        "category_plan": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_category_compute_plan.csv", rows)
    write_csv(out_dir / "table_parallel_scenarios.csv", scenarios)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# DiffusionAD Measured Compute Plan",
        "",
        f"- Timing: `{seconds_per_batch:.2f}` seconds per effective batch.",
        f"- Full matrix: `{summary['optimizer_steps']}` optimizer steps.",
        f"- Aggregate compute: `{total_gpu_hours:.1f}` GPU hours "
        f"(`{total_gpu_hours / 24.0:.1f}` GPU days).",
        f"- Checkpoint storage: "
        f"`{summary['estimated_checkpoint_storage_gib']:.1f} GiB`.",
        "- Stable local mode: author batch 16, micro-batch 4, one optimizer "
        "step per author batch, CUDA AMP.",
        "- BatchNorm uses micro-batch statistics and must be disclosed.",
        "",
        "## Parallel Scenarios",
        "",
        "| GPUs | Ideal greedy wall hours | Ideal greedy wall days |",
        "|---:|---:|---:|",
    ]
    for scenario in scenarios:
        lines.append(
            f"| {scenario['gpu_count']} | "
            f"{scenario['ideal_greedy_wall_hours']:.1f} | "
            f"{scenario['ideal_greedy_wall_days']:.1f} |"
        )
    lines.extend(
        [
            "",
            "These are compute-only estimates from a measured 10-epoch "
            "segment. Evaluation, asset sync, queueing, and checkpoint "
            "transfer add wall time.",
            "",
        ]
    )
    (out_dir / "compute_plan.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
