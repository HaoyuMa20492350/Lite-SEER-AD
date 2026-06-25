from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export qualitative case artifacts from a run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=12)
    return p.parse_args()


def _ranked_image_dirs(run_dir: Path, limit: int) -> list[Path]:
    scores_path = run_dir / "scores.csv"
    image_root = run_dir / "images"
    if not scores_path.exists():
        return sorted(image_root.glob("*"))[:limit]
    normal: list[tuple[float, Path]] = []
    anomaly: list[tuple[float, Path]] = []
    with scores_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(float(row.get("index", 0)))
            score = float(row.get("image_score", 0.0))
            image_dir = image_root / f"{idx:05d}"
            if not image_dir.exists():
                continue
            if int(float(row.get("label", 0))) == 0:
                normal.append((score, image_dir))
            else:
                anomaly.append((score, image_dir))
    normal = sorted(normal, key=lambda x: x[0], reverse=True)
    anomaly = sorted(anomaly, key=lambda x: x[0], reverse=True)
    selected: list[Path] = []
    half = max(1, limit // 2)
    selected.extend([p for _, p in normal[:half]])
    selected.extend([p for _, p in anomaly[: max(0, limit - len(selected))]])
    if len(selected) < limit:
        for _, path in normal[half:] + anomaly:
            if path not in selected:
                selected.append(path)
            if len(selected) >= limit:
                break
    return selected[:limit]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "qualitative_cases"
    out.mkdir(parents=True, exist_ok=True)
    image_dirs = _ranked_image_dirs(run_dir, args.limit)
    copied = 0
    for image_dir in image_dirs:
        dest = out / image_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in [
            "input.png",
            "reconstruction.png",
            "residual.png",
            "candidate_roi.png",
            "verified_roi.png",
            "mask.png",
            "ground_truth.png",
            "final_heatmap.png",
            "final_mask.png",
            "repair.png",
            "roi_log.jsonl",
            "residual_heatmap.npz",
        ]:
            src = image_dir / name
            if src.exists():
                shutil.copy2(src, dest / name)
                copied += 1
    print(f"Copied {copied} qualitative artifacts to {out}")


if __name__ == "__main__":
    main()
