from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


IMAGE_COLUMNS = [
    ("input.png", "Input"),
    ("candidate_roi.png", "Candidate ROI"),
    ("reconstruction.png", "Reconstruction"),
    ("repair.png", "Local repair"),
    ("residual.png", "Residual"),
    ("final_heatmap.png", "Final heatmap"),
    ("ground_truth.png", "Ground truth"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a repair-process visualization panel.")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rows", type=int, default=4)
    return p.parse_args()


def read_scores(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def best_case(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    qualitative = run_dir / "qualitative_cases"
    candidates = [path for path in qualitative.iterdir() if path.is_dir()] if qualitative.exists() else []
    if not candidates:
        images = run_dir / "images"
        candidates = [path for path in images.iterdir() if path.is_dir()] if images.exists() else []
    complete = [
        path
        for path in candidates
        if all((path / filename).exists() for filename, _label in IMAGE_COLUMNS)
    ]
    if not complete:
        return None
    score_by_index = {
        int(float(row.get("index", 0))): row for row in read_scores(run_dir / "scores.csv")
    }
    ranked = []
    for path in complete:
        try:
            index = int(path.name)
        except ValueError:
            continue
        score = score_by_index.get(index, {})
        label = int(float(score.get("label", 0)))
        image_score = float(score.get("image_score", 0.0))
        ranked.append((label, image_score, path, score))
    if not ranked:
        return complete[0], {}
    _label, _score, path, metadata = max(ranked, key=lambda item: (item[0], item[1]))
    return path, metadata


def category_from_run(run_dir: Path) -> str:
    payload_path = run_dir / "run_args.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        args = payload.get("args", {})
        if isinstance(args, dict) and args.get("category"):
            return str(args["category"])
    return run_dir.name


def main() -> None:
    args = parse_args()
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export the repair panel") from exc

    selected = []
    for run_name in args.run:
        run_dir = Path(run_name)
        case = best_case(run_dir)
        if case is None:
            continue
        case_dir, metadata = case
        selected.append((run_dir, case_dir, metadata))
        if len(selected) >= args.max_rows:
            break
    if not selected:
        raise FileNotFoundError("No run contains a complete repair visualization case")

    cell_w = 180
    image_h = 150
    row_h = image_h + 42
    header_h = 34
    panel = Image.new(
        "RGB",
        (cell_w * len(IMAGE_COLUMNS), header_h + row_h * len(selected)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    for column, (_filename, label) in enumerate(IMAGE_COLUMNS):
        draw.text((column * cell_w + 8, 10), label, fill=(0, 0, 0))

    index_rows = []
    for row_index, (run_dir, case_dir, metadata) in enumerate(selected):
        y0 = header_h + row_index * row_h
        title = f"{category_from_run(run_dir)} | sample {case_dir.name}"
        draw.text((8, y0 + 5), title, fill=(0, 0, 0))
        for column, (filename, _label) in enumerate(IMAGE_COLUMNS):
            image = Image.open(case_dir / filename).convert("RGB")
            image.thumbnail((cell_w - 12, image_h - 8))
            x = column * cell_w + (cell_w - image.width) // 2
            y = y0 + 30 + (image_h - image.height) // 2
            panel.paste(image, (x, y))
            draw.rectangle((x, y, x + image.width, y + image.height), outline=(170, 170, 170))
        index_rows.append(
            {
                "run": str(run_dir),
                "category": category_from_run(run_dir),
                "case": case_dir.name,
                "path": metadata.get("path", ""),
                "label": metadata.get("label", ""),
                "image_score": metadata.get("image_score", ""),
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.save(out_dir / "fig_repair_process_panel.png")
    with (out_dir / "table_repair_visualization_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "category", "case", "path", "label", "image_score"],
        )
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(index_rows)} repair cases to {out_dir}")


if __name__ == "__main__":
    main()
