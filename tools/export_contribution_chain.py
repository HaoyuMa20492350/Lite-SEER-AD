from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export the feature-first contribution chain table and figure."
    )
    p.add_argument(
        "--paper-package",
        default="tables/feature_first_fusion_aggregate_paper_package",
    )
    p.add_argument(
        "--stability",
        default="tables/synthetic_gate_fusion_aggregate_stability/table_delta_mean_std.csv",
    )
    p.add_argument(
        "--out",
        default="tables/feature_first_fusion_aggregate_paper_package",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    args = parse_args()
    package = Path(args.paper_package)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = read_csv(package / "table_paired_bootstrap_ci.csv")
    modules = read_csv(package / "table_module_ablation_mpdd.csv")
    stability = read_csv(Path(args.stability))
    summary = json.loads(
        (package / "summary.json").read_text(encoding="utf-8")
    )
    repair_summary_path = package / "repair_quality_summary.json"
    repair_summary = (
        json.loads(repair_summary_path.read_text(encoding="utf-8"))
        if repair_summary_path.exists()
        else {}
    )
    module_summary_path = package / "module_evidence_summary.json"
    module_summary = (
        json.loads(module_summary_path.read_text(encoding="utf-8"))
        if module_summary_path.exists()
        else {}
    )
    repair_sdr = repair_summary.get("sdr_gt", {})
    repair_downgraded = (
        repair_summary.get("claim_decision") == "downgrade_to_visualization_only"
    )

    overall = {
        row["metric"]: row
        for row in bootstrap
        if row["dataset"] == "all"
    }
    std_by_metric = {
        row["metric"]: as_float(row["std"]) for row in stability
    }
    hn = next(row for row in modules if row["module"] == "HN-SEV")
    crv = next(row for row in modules if row["module"] == "CRV")
    lc_rows = [row for row in modules if row["module"] == "LC-RDS"]
    latency_values = [
        as_float(row["mean_delta_latency_ms"]) for row in lc_rows
    ]

    nodes = [
        {
            "order": 1,
            "component": "Feature prior",
            "role": "Frozen detection and localization core",
            "evidence": (
                f"Pixel AP delta {as_float(overall['pixel_ap']['mean_delta']):+.4f}; "
                f"95% CI [{as_float(overall['pixel_ap']['ci95_low']):+.4f}, "
                f"{as_float(overall['pixel_ap']['ci95_high']):+.4f}]"
            ),
            "claim": "Main detector",
        },
        {
            "order": 2,
            "component": "Cross-seed synthetic gate",
            "role": "Label-free candidate selection",
            "evidence": (
                f"{100.0 * as_float(summary['category_selection_agreement']):.1f}% "
                f"agreement; AP/Dice std "
                f"{std_by_metric['delta_pixel_ap']:.4f}/"
                f"{std_by_metric['delta_dice']:.4f}"
            ),
            "claim": "Selection stability",
        },
        {
            "order": 3,
            "component": "Normal-calibrated fusion",
            "role": "Complementary localization candidate",
            "evidence": (
                f"AP wins {overall['pixel_ap']['wins']}/33; "
                f"Dice wins {overall['dice']['wins']}/33; "
                f"AUPRO wins {overall['aupro']['wins']}/33"
            ),
            "claim": "Localization gain",
        },
        {
            "order": 4,
            "component": "HN-SEV",
            "role": "Hard-negative verification filter",
            "evidence": (
                (
                    f"FPRR delta "
                    f"{as_float(module_summary['hn_sev']['mean_delta_fprr']):+.4f}; "
                    f"{module_summary['hn_sev']['categories_with_lower_fprr']}/33 "
                    "categories lower"
                )
                if module_summary
                else f"FPRR delta {as_float(hn['mean_delta_fprr']):+.4f}"
            ),
            "claim": "Verification evidence",
        },
        {
            "order": 5,
            "component": "CRV",
            "role": "Multi-space repair diagnostic",
            "evidence": (
                (
                    f"33-category SDR-GT Spearman "
                    f"{as_float(repair_sdr.get('sdr_gt_fraction_spearman')):+.4f}; "
                    f"{repair_sdr.get('roi_count', 0)} ROIs; detector frozen"
                )
                if repair_summary
                else (
                    f"RDC delta {as_float(crv['mean_delta_rdc']):+.4f}; "
                    f"SDR delta {as_float(crv['mean_delta_sdr_mean']):+.4f}; "
                    "detector metrics frozen"
                )
            ),
            "claim": (
                "Visualization only; no GT-alignment claim"
                if repair_downgraded
                else "Explanation, not AP/Dice gain"
            ),
        },
        {
            "order": 6,
            "component": "LC-RDS",
            "role": "Repair-gain/latency scheduler",
            "evidence": (
                (
                    "Latency vs fixed25/rule "
                    f"{as_float(module_summary['lc_rds']['fixed25']['mean_delta_latency_ms']):+.1f}/"
                    f"{as_float(module_summary['lc_rds']['rule']['mean_delta_latency_ms']):+.1f} "
                    "ms; both 33/33 faster"
                )
                if module_summary
                else (
                    "Latency deltas "
                    + "/".join(f"{value:+.1f}" for value in latency_values)
                    + " ms"
                )
            ),
            "claim": "Efficiency evidence",
        },
    ]
    with (out_dir / "table_contribution_chain.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order", "component", "role", "evidence", "claim"],
        )
        writer.writeheader()
        writer.writerows(nodes)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export the contribution figure") from exc

    width, height = 1800, 680
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    title_font = ImageFont.truetype(str(bold_path), 32)
    node_font = ImageFont.truetype(str(bold_path), 22)
    body_font = ImageFont.truetype(str(font_path), 17)
    small_font = ImageFont.truetype(str(font_path), 15)

    draw.text(
        (52, 34),
        "Lite-SEER-AD Feature-First Contribution Chain",
        fill="#172033",
        font=title_font,
    )
    draw.text(
        (52, 78),
        "Detection remains frozen; verification and repair modules provide separate evidence.",
        fill="#526074",
        font=body_font,
    )

    box_w, box_h = 360, 170
    positions = [
        (60, 150),
        (490, 150),
        (920, 150),
        (490, 430),
        (920, 430),
        (1350, 430),
    ]
    colors = [
        ("#e7f0ff", "#2b66b1"),
        ("#e9f7ef", "#287a4b"),
        ("#fff4dd", "#a96800"),
        ("#f2ecff", "#6545a6"),
        ("#fcebed", "#a23d4c"),
        ("#e9f4f5", "#33727a"),
    ]

    def arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line([start, end], fill="#7b8797", width=4)
        x, y = end
        draw.polygon(
            [(x, y), (x - 14, y - 8), (x - 14, y + 8)],
            fill="#7b8797",
        )

    arrow((420, 235), (490, 235))
    arrow((850, 235), (920, 235))
    draw.line([(670, 320), (670, 390)], fill="#7b8797", width=4)
    draw.line([(670, 390), (490, 390)], fill="#7b8797", width=4)
    arrow((490, 390), (490, 515))
    arrow((850, 515), (920, 515))
    arrow((1280, 515), (1350, 515))

    for node, (x, y), (fill, border) in zip(nodes, positions, colors):
        draw.rounded_rectangle(
            (x, y, x + box_w, y + box_h),
            radius=6,
            fill=fill,
            outline=border,
            width=3,
        )
        draw.text(
            (x + 20, y + 18),
            str(node["component"]),
            fill="#172033",
            font=node_font,
        )
        draw.text(
            (x + 20, y + 54),
            str(node["role"]),
            fill="#354157",
            font=body_font,
        )
        evidence = str(node["evidence"])
        words = evidence.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=small_font) <= box_w - 40:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for index, line in enumerate(lines[:3]):
            draw.text(
                (x + 20, y + 88 + 23 * index),
                line,
                fill="#526074",
                font=small_font,
            )
        draw.text(
            (x + 20, y + box_h - 28),
            f"Claim: {node['claim']}",
            fill=border,
            font=small_font,
        )

    image.save(out_dir / "fig_contribution_chain.png")
    print(f"Wrote contribution chain to {out_dir}")


if __name__ == "__main__":
    main()
