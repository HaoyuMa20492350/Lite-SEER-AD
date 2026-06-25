from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
MAIN_FIELDS = ["dataset", "source", "method", "category", *METRICS]
EFF_FIELDS = ["dataset", "source", "method", "category", "latency_ms_mean", "fps", "nfe_mean"]
GATE_FIELDS = ["dataset", "phase", "module", "passes", "total", "ready"]
CATEGORY_DELTA_FIELDS = [
    "dataset",
    "category",
    "ours_method",
    "ours_pixel_ap",
    "best_baseline",
    "best_pixel_ap",
    "pixel_ap_delta",
    "ours_aupro",
    "best_aupro",
    "aupro_delta",
]
QUAL_FIELDS = ["dataset", "run", "category", "path", "png_count"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper-facing cross-dataset evidence package tables.")
    p.add_argument("--out", default="tables/paper_package")
    p.add_argument("--ours-method", default="lite_seer_ad_crv035")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def add_main(rows: list[dict[str, Any]], dataset: str, source: str, method: str | None, data: list[dict[str, str]]) -> None:
    for row in data:
        out = {"dataset": dataset, "source": source, "method": method or row.get("method", ""), "category": row.get("category", "")}
        for metric in METRICS:
            out[metric] = row.get(metric, "")
        rows.append(out)


def add_eff(rows: list[dict[str, Any]], dataset: str, source: str, method: str | None, data: list[dict[str, str]]) -> None:
    for row in data:
        out = {
            "dataset": dataset,
            "source": source,
            "method": method or row.get("method", ""),
            "category": row.get("category", ""),
            "latency_ms_mean": row.get("latency_ms_mean", ""),
            "fps": row.get("fps", ""),
            "nfe_mean": row.get("nfe_mean", ""),
        }
        rows.append(out)


def add_gate(rows: list[dict[str, Any]], dataset: str, phase: str, gate: dict[str, Any]) -> None:
    passes = gate.get("module_passes", {})
    totals = gate.get("module_total", {})
    ready = gate.get("module_ready", {})
    for module in sorted(set(passes) | set(totals) | set(ready)):
        rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "module": module,
                "passes": passes.get(module, ""),
                "total": totals.get(module, ""),
                "ready": ready.get(module, ""),
            }
        )


def coverage_row(dataset: str, path: Path) -> dict[str, Any]:
    coverage = read_json(path)
    return {
        "dataset": dataset,
        "required_methods": ",".join(coverage.get("required_methods", [])),
        "categories": len(coverage.get("categories", [])),
        "present": coverage.get("present", ""),
        "total_required": coverage.get("total_required", ""),
        "missing": coverage.get("missing", ""),
        "complete": coverage.get("complete", ""),
    }


def report_row(name: str, expected: int | None = None) -> dict[str, Any]:
    report = read_json(Path("runs") / f"{name}_report.json")
    schema = report.get("schema", [])
    failures = report.get("failures", [])
    return {
        "name": name,
        "schema_runs": len(schema),
        "expected_runs": expected if expected is not None else "",
        "schema_ok": all(bool(row.get("ok")) for row in schema) if schema else False,
        "failures": len(failures),
    }


def means_by_dataset_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("source", "")), str(row.get("method", "")))
        grouped.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = fnum(row.get(metric))
            if value == value:
                grouped[key][metric].append(value)
    out = []
    for (dataset, source, method), values in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "source": source, "method": method}
        for metric, vals in values.items():
            row[metric] = mean(vals) if vals else ""
        out.append(row)
    return out


def category_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("category", "")))
        by_group.setdefault(key, []).append(row)
    out = []
    ours_methods = {"lite_seer_ad_crv035", "lite_seer_ad"}
    excluded = ours_methods | {"lite_seer_ad_crv05"}
    for (dataset, category), items in sorted(by_group.items()):
        ours = next((row for row in items if str(row.get("method", "")) in ours_methods), None)
        baselines = [row for row in items if str(row.get("method", "")) not in excluded]
        baselines = [row for row in baselines if fnum(row.get("pixel_ap")) == fnum(row.get("pixel_ap"))]
        if not ours or not baselines:
            continue
        best_pixel = max(baselines, key=lambda row: fnum(row.get("pixel_ap")))
        best_aupro = max(baselines, key=lambda row: fnum(row.get("aupro")))
        out.append(
            {
                "dataset": dataset,
                "category": category,
                "ours_method": ours.get("method", ""),
                "ours_pixel_ap": ours.get("pixel_ap", ""),
                "best_baseline": best_pixel.get("method", ""),
                "best_pixel_ap": best_pixel.get("pixel_ap", ""),
                "pixel_ap_delta": fnum(ours.get("pixel_ap")) - fnum(best_pixel.get("pixel_ap")),
                "ours_aupro": ours.get("aupro", ""),
                "best_aupro": best_aupro.get("aupro", ""),
                "aupro_delta": fnum(ours.get("aupro")) - fnum(best_aupro.get("aupro")),
            }
        )
    return out


def qualitative_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path("runs").glob("*/qualitative_cases")):
        run = path.parent.name
        if run.startswith("visa_"):
            dataset = "visa"
        elif run.startswith("mpdd_"):
            dataset = "mpdd"
        elif run.startswith("mini_mvtec") or run.startswith("next_models") or run.startswith("mvtec"):
            dataset = "mvtec"
        else:
            continue
        category = run
        for prefix in ["visa_mini_", "visa_gate_", "mpdd_mini_", "mpdd_gate_", "mini_mvtec_long_", "mini_mvtec_"]:
            if category.startswith(prefix):
                category = category[len(prefix) :]
                break
        if category.endswith("_full"):
            category = category[: -len("_full")]
        rows.append(
            {
                "dataset": dataset,
                "run": run,
                "category": category,
                "path": path.as_posix(),
                "png_count": len(list(path.rglob("*.png"))),
            }
        )
    return rows


def write_pareto_plot(out: Path, main_means: list[dict[str, Any]], eff_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    eff_means = means_efficiency_by_dataset_method(eff_rows)
    eff_lookup = {
        (row.get("dataset", ""), row.get("source", ""), row.get("method", "")): row for row in eff_means
    }
    datasets = ["mvtec15", "visa", "mpdd"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        ax.set_title(dataset)
        ax.set_xlabel("FPS")
        ax.set_ylabel("Pixel AP")
        ax.set_xscale("log")
        for row in main_means:
            if row.get("dataset") != dataset:
                continue
            eff = eff_lookup.get((row.get("dataset"), row.get("source"), row.get("method")), {})
            fps = fnum(eff.get("fps"))
            pixel_ap = fnum(row.get("pixel_ap"))
            if fps != fps or pixel_ap != pixel_ap or fps <= 0:
                continue
            method = str(row.get("method", ""))
            marker = "*" if method.startswith("lite_seer") or method == "lite_seer_ad" else "o"
            ax.scatter(fps, pixel_ap, marker=marker)
            ax.annotate(method.replace("lite_seer_ad", "ours"), (fps, pixel_ap), fontsize=7)
        ax.grid(True, alpha=0.25)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_pareto_pixel_ap_fps.png", dpi=180)
    plt.close(fig)


def _candidate_run_dirs(dataset: str, category: str) -> list[Path]:
    if dataset == "visa":
        return [Path("runs") / f"visa_mini_{category}_full", Path("runs") / f"visa_gate_{category}_full"]
    if dataset == "mpdd":
        return [Path("runs") / f"mpdd_mini_{category}_full", Path("runs") / f"mpdd_gate_{category}_full"]
    if dataset == "mvtec15":
        return [
            Path("runs") / f"mini_mvtec_long_{category}_full",
            Path("runs") / f"mini_mvtec_{category}_full",
        ]
    return []


def write_failure_panel(out: Path, deltas: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    image_names = ["input.png", "final_heatmap.png", "ground_truth.png", "final_mask.png"]
    selected: list[tuple[str, str, Path]] = []
    for row in sorted(deltas, key=lambda item: fnum(item.get("pixel_ap_delta"))):
        dataset = str(row.get("dataset", ""))
        if any(existing[0] == dataset for existing in selected):
            continue
        category = str(row.get("category", ""))
        for run_dir in _candidate_run_dirs(dataset, category):
            qual = run_dir / "qualitative_cases"
            if qual.exists():
                case_dirs = sorted([p for p in qual.iterdir() if p.is_dir()])
                case_dir = next((p for p in case_dirs if all((p / name).exists() for name in image_names)), None)
                if case_dir is not None:
                    selected.append((dataset, category, case_dir))
                    break
        if len(selected) >= 3:
            break
    if not selected:
        return

    cell_w, cell_h = 160, 188
    title_h = 28
    panel = Image.new("RGB", (cell_w * len(image_names), title_h + cell_h * len(selected)), "white")
    draw = ImageDraw.Draw(panel)
    for col, name in enumerate(["input", "heatmap", "ground truth", "prediction"]):
        draw.text((col * cell_w + 8, 8), name, fill=(0, 0, 0))
    for row_idx, (dataset, category, case_dir) in enumerate(selected):
        y0 = title_h + row_idx * cell_h
        draw.text((8, y0 + 4), f"{dataset}: {category}", fill=(0, 0, 0))
        for col, image_name in enumerate(image_names):
            image_path = case_dir / image_name
            if not image_path.exists():
                continue
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((cell_w - 16, cell_h - 36))
            x = col * cell_w + (cell_w - img.width) // 2
            y = y0 + 28 + (cell_h - 36 - img.height) // 2
            panel.paste(img, (x, y))
            draw.rectangle((x, y, x + img.width, y + img.height), outline=(180, 180, 180))
    out.mkdir(parents=True, exist_ok=True)
    panel.save(out / "fig_qualitative_failure_panel.png")


def means_efficiency_by_dataset_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["latency_ms_mean", "fps", "nfe_mean"]
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("source", "")), str(row.get("method", "")))
        grouped.setdefault(key, {field: [] for field in fields})
        for field in fields:
            value = fnum(row.get(field))
            if value == value:
                grouped[key][field].append(value)
    out = []
    for (dataset, source, method), values in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "source": source, "method": method}
        for field, vals in values.items():
            row[field] = mean(vals) if vals else ""
        out.append(row)
    return out


def write_notes(out: Path, status_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Lite-SEER-AD Paper Package Notes",
        "",
        "## Scope",
        "",
        "- This package merges current MVTec15, VisA, and MPDD evidence into paper-facing CSV tables.",
        "- VisA/MPDD baseline tables are now full-category local baseline runs for six methods.",
        "- MVTec15 remains the strongest comparison anchor; VisA/MPDD results are cross-dataset evidence and failure-analysis material.",
        "",
        "## Run Status",
        "",
    ]
    for row in status_rows:
        lines.append(
            f"- {row['name']}: schema `{row['schema_runs']}/{row['expected_runs']}`, "
            f"schema_ok=`{row['schema_ok']}`, failures=`{row['failures']}`."
        )
    lines.extend(["", "## Baseline Coverage", ""])
    for row in coverage_rows:
        lines.append(
            f"- {row['dataset']}: present `{row['present']}/{row['total_required']}`, "
            f"missing=`{row['missing']}`, complete=`{row['complete']}`."
        )
    lines.extend(
        [
            "",
            "## Failure-Analysis Position",
            "",
            "- The current evidence does not support a broad SOTA claim across datasets.",
            "- The safer paper route is repair-aware verification, module behavior, cross-dataset stress testing, and transparent failure cases.",
        "- Use category-level tables to identify reflective/low-contrast classes where Lite-SEER-AD underperforms memory baselines.",
        "- Use `table_category_deltas.csv` to select failure examples and `table_qualitative_case_index.csv` to locate existing visual panels.",
        "- Use `fig_pareto_pixel_ap_fps.png` for a compact speed/quality view.",
        "- Use `fig_qualitative_failure_panel.png` as a first failure-panel draft.",
        "",
    ]
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_analysis_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    main_rows: list[dict[str, Any]] = []
    eff_rows: list[dict[str, Any]] = []

    add_main(main_rows, "mvtec15", "comparison", None, read_csv(Path("tables/mvtec15_comparison/table_main_ours_vs_baselines.csv")))
    add_eff(eff_rows, "mvtec15", "comparison", None, read_csv(Path("tables/mvtec15_comparison/table_efficiency_ours_vs_baselines.csv")))

    add_main(main_rows, "visa", "ours_mini", "lite_seer_ad", read_csv(Path("tables/visa_mini/table_main_visa.csv")))
    add_eff(eff_rows, "visa", "ours_mini", "lite_seer_ad", read_csv(Path("tables/visa_mini/table_efficiency_visa.csv")))
    add_main(main_rows, "visa", "baseline", None, read_csv(Path("tables/visa_baselines/table_main_visa.csv")))
    add_eff(eff_rows, "visa", "baseline", None, read_csv(Path("tables/visa_baselines/table_efficiency_visa.csv")))

    add_main(main_rows, "mpdd", "ours_mini", "lite_seer_ad", read_csv(Path("tables/mpdd_mini/table_main_mpdd.csv")))
    add_eff(eff_rows, "mpdd", "ours_mini", "lite_seer_ad", read_csv(Path("tables/mpdd_mini/table_efficiency_mpdd.csv")))
    add_main(main_rows, "mpdd", "baseline", None, read_csv(Path("tables/mpdd_baselines/table_main_mpdd.csv")))
    add_eff(eff_rows, "mpdd", "baseline", None, read_csv(Path("tables/mpdd_baselines/table_efficiency_mpdd.csv")))

    gate_rows: list[dict[str, Any]] = []
    add_gate(gate_rows, "mvtec15", "ours", read_json(Path("tables/mvtec15_ours/gate_summary.json")))
    add_gate(gate_rows, "visa", "mini", read_json(Path("tables/visa_mini/gate_summary.json")))
    add_gate(gate_rows, "visa", "gate", read_json(Path("tables/visa_gate/gate_summary.json")))
    add_gate(gate_rows, "mpdd", "mini", read_json(Path("tables/mpdd_mini/gate_summary.json")))
    add_gate(gate_rows, "mpdd", "gate", read_json(Path("tables/mpdd_gate/gate_summary.json")))

    status_rows = [
        report_row("visa_mini", 60),
        report_row("visa_gate", 72),
        report_row("mpdd_mini", 30),
        report_row("mpdd_gate", 36),
        report_row("visa_baseline", 72),
        report_row("mpdd_baseline", 36),
    ]
    coverage_rows = [
        coverage_row("mvtec15", Path("tables/mvtec15_baselines/baseline_coverage.json")),
        coverage_row("visa", Path("tables/visa_baselines/baseline_coverage.json")),
        coverage_row("mpdd", Path("tables/mpdd_baselines/baseline_coverage.json")),
    ]

    write_csv(out / "table_main_cross_dataset.csv", main_rows, MAIN_FIELDS)
    write_csv(out / "table_efficiency_cross_dataset.csv", eff_rows, EFF_FIELDS)
    main_means = means_by_dataset_method(main_rows)
    write_csv(out / "table_mean_by_dataset_method.csv", main_means, ["dataset", "source", "method", *METRICS])
    deltas = category_delta_rows(main_rows)
    write_csv(out / "table_category_deltas.csv", deltas, CATEGORY_DELTA_FIELDS)
    write_csv(out / "table_module_gates.csv", gate_rows, GATE_FIELDS)
    write_csv(out / "table_run_status.csv", status_rows, ["name", "schema_runs", "expected_runs", "schema_ok", "failures"])
    write_csv(out / "table_baseline_coverage.csv", coverage_rows, ["dataset", "required_methods", "categories", "present", "total_required", "missing", "complete"])
    write_csv(out / "table_qualitative_case_index.csv", qualitative_rows(), QUAL_FIELDS)
    write_pareto_plot(out, main_means, eff_rows)
    write_failure_panel(out, deltas)
    write_notes(out, status_rows, coverage_rows)
    print(f"Wrote paper package to {out}")


if __name__ == "__main__":
    main()
