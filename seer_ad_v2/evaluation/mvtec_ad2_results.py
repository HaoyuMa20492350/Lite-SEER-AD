from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from seer_ad_v2.evaluation.mvtec_ad2_pipeline import (
    DEFAULT_SEEDS,
    OFFICIAL_CATEGORIES,
)


CORE_METRICS = ("image_auroc", "pixel_auroc", "aupro")
OPTIONAL_METRICS = ("pixel_ap", "dice", "f1")
METRIC_ALIASES = {
    "image_auroc": "image_auroc",
    "image-auroc": "image_auroc",
    "i-auroc": "image_auroc",
    "image_roc_auc": "image_auroc",
    "pixel_auroc": "pixel_auroc",
    "pixel-auroc": "pixel_auroc",
    "p-auroc": "pixel_auroc",
    "pixel_roc_auc": "pixel_auroc",
    "aupro": "aupro",
    "au-pro": "aupro",
    "au_pro": "aupro",
    "pixel_ap": "pixel_ap",
    "pixel-ap": "pixel_ap",
    "p-ap": "pixel_ap",
    "dice": "dice",
    "f1": "f1",
}
OFFICIAL_HOSTS = {"benchmark.mvtec.com", "www.benchmark.mvtec.com"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_official_server_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "") in OFFICIAL_HOSTS


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid metric value: {value!r}") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Metric value must be finite and in [0, 1]: {value!r}")
    return result


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {_normalized_key(str(key)): value for key, value in row.items()}
    result: dict[str, Any] = {
        "split": str(normalized.get("split", "test_private")).strip(),
        "category": str(normalized.get("category", "all")).strip(),
    }
    for key, value in normalized.items():
        metric = METRIC_ALIASES.get(key)
        if metric is not None and value not in (None, ""):
            result[metric] = _as_float(value)
    missing = [metric for metric in CORE_METRICS if metric not in result]
    if missing:
        raise ValueError(f"Official result row is missing core metrics: {missing}")
    if result["split"] not in {
        "test_private",
        "test_private_mixed",
        "private_overall",
    }:
        raise ValueError(f"Unsupported official private split: {result['split']}")
    if not result["category"]:
        raise ValueError("Official result category cannot be empty.")
    return result


def load_result_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict):
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"results", "rows"}
            }
            raw_rows = payload.get("results", payload.get("rows", []))
        else:
            raise ValueError("Official result JSON must be an object or list.")
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Official result export must be JSON or CSV.")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Official result export contains no result rows.")
    return metadata, [normalize_result_row(dict(row)) for row in raw_rows]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_evaluation_metadata(
    metadata: dict[str, Any],
    *,
    source_url: str | None = None,
    submission_id: str | None = None,
    evaluated_at: str | None = None,
) -> dict[str, str]:
    values = {
        "source_url": str(source_url or metadata.get("source_url", "")).strip(),
        "submission_id": str(
            submission_id or metadata.get("submission_id", "")
        ).strip(),
        "evaluated_at": str(
            evaluated_at or metadata.get("evaluated_at", "")
        ).strip(),
    }
    if not is_official_server_url(values["source_url"]):
        raise ValueError(
            "Official evaluation source must be https://benchmark.mvtec.com/."
        )
    if not values["submission_id"]:
        raise ValueError("Official evaluation submission_id is required.")
    try:
        datetime.fromisoformat(values["evaluated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Official evaluation evaluated_at must be ISO-8601."
        ) from exc
    return values


def validate_local_submission_evidence(
    protocol: dict[str, Any],
    archive_path: Path,
) -> None:
    checker_status = (protocol.get("official_checker") or {}).get("status")
    if protocol.get("files") != 4090:
        raise ValueError("Local private submission must contain 4090 samples.")
    if protocol.get("full_official_submission") is not True:
        raise ValueError("Local submission is not marked as full official.")
    if checker_status != "passed":
        raise ValueError("Local submission has not passed the official checker.")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    protocol_archive = protocol.get("archive")
    if (
        protocol_archive
        and Path(str(protocol_archive)).resolve() != archive_path.resolve()
    ):
        raise ValueError(
            "Local archive path does not match the submission protocol."
        )
    protocol_size = protocol.get("archive_size_bytes")
    if (
        protocol_size is not None
        and int(protocol_size) != archive_path.stat().st_size
    ):
        raise ValueError(
            "Local archive size does not match the submission protocol."
        )
    protocol_sha256 = protocol.get("archive_sha256")
    if protocol_sha256 and str(protocol_sha256).lower() != sha256_file(
        archive_path
    ).lower():
        raise ValueError(
            "Local archive SHA256 does not match the submission protocol."
        )


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    return {
        "rows": len(items),
        "splits": sorted({str(row["split"]) for row in items}),
        "categories": sorted({str(row["category"]) for row in items}),
        "means": {
            metric: sum(float(row[metric]) for row in items) / len(items)
            for metric in (*CORE_METRICS, *OPTIONAL_METRICS)
            if items and all(metric in row for row in items)
        },
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["split", "category", *CORE_METRICS, *OPTIONAL_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_public_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items = [dict(row) for row in rows]
    expected = {
        (category, seed)
        for category in OFFICIAL_CATEGORIES
        for seed in DEFAULT_SEEDS
    }
    observed = set()
    normalized = []
    for row in items:
        category = str(row.get("category", "")).strip()
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MVTec AD 2 public seed: {row.get('seed')}") from exc
        observed.add((category, seed))
        metrics = {
            metric: _as_float(row.get(metric))
            for metric in CORE_METRICS
        }
        for metric in OPTIONAL_METRICS:
            if row.get(metric) not in (None, ""):
                metrics[metric] = _as_float(row.get(metric))
        normalized.append(
            {"category": category, "seed": seed, **metrics}
        )
    if len(items) != len(expected) or observed != expected:
        raise ValueError(
            "MVTec AD 2 public table must contain exactly 8 categories x "
            "3 seeds."
        )
    output = []
    for seed in DEFAULT_SEEDS:
        seed_rows = [row for row in normalized if row["seed"] == seed]
        output.append(
            {
                "dataset": "mvtec_ad2",
                "evaluation_scope": "test_public",
                "evaluation_source": "local_public_ground_truth",
                "seed": seed,
                "categories": len(seed_rows),
                **{
                    metric: sum(float(row[metric]) for row in seed_rows)
                    / len(seed_rows)
                    for metric in (*CORE_METRICS, *OPTIONAL_METRICS)
                    if all(metric in row for row in seed_rows)
                },
            }
        )
    return output


def combined_results_rows(
    existing_rows: Iterable[dict[str, Any]],
    public_rows: Iterable[dict[str, Any]],
    evaluation: dict[str, Any],
    *,
    private_seed: int = 7,
) -> list[dict[str, Any]]:
    combined = []
    for row in existing_rows:
        combined.append(
            {
                "dataset": row.get("dataset"),
                "evaluation_scope": "heldout_test",
                "evaluation_source": "local_path_aligned",
                "seed": row.get("seed"),
                "categories": row.get("categories"),
                **{
                    metric: row.get(metric)
                    for metric in (*CORE_METRICS, *OPTIONAL_METRICS)
                    if row.get(metric) not in (None, "")
                },
            }
        )
    combined.extend(aggregate_public_rows(public_rows))
    private_means = evaluation["summary"]["means"]
    combined.append(
        {
            "dataset": "mvtec_ad2",
            "evaluation_scope": "test_private_and_mixed",
            "evaluation_source": "official_benchmark_server",
            "seed": private_seed,
            "categories": len(OFFICIAL_CATEGORIES),
            **private_means,
        }
    )
    return combined


def normalize_leaderboard_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {_normalized_key(str(key)): value for key, value in row.items()}
    method = str(
        normalized.get("method", normalized.get("method_name", ""))
    ).strip()
    submission_id = str(normalized.get("submission_id", "")).strip()
    if not method or not submission_id:
        raise ValueError(
            "Leaderboard rows require method and submission_id."
        )
    result = {"method": method, "submission_id": submission_id}
    for metric in CORE_METRICS:
        aliases = [
            alias for alias, canonical in METRIC_ALIASES.items()
            if canonical == metric
        ]
        value = next(
            (
                normalized[alias]
                for alias in aliases
                if normalized.get(alias) not in (None, "")
            ),
            None,
        )
        if value is None:
            raise ValueError(
                f"Leaderboard row is missing core metric {metric}."
            )
        result[metric] = _as_float(value)
    return result


def load_leaderboard_rows(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata: dict[str, Any] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict):
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"leaderboard", "results", "rows"}
            }
            raw_rows = payload.get(
                "leaderboard",
                payload.get("results", payload.get("rows", [])),
            )
        else:
            raise ValueError("Leaderboard JSON must be an object or list.")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Leaderboard snapshot must be JSON or CSV.")
    if not isinstance(raw_rows, list) or len(raw_rows) < 2:
        raise ValueError("Leaderboard snapshot must contain at least two methods.")
    rows = [normalize_leaderboard_row(dict(row)) for row in raw_rows]
    ids = [row["submission_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Leaderboard submission IDs must be unique.")
    return metadata, rows


def evaluate_leaderboard(
    evaluation: dict[str, Any],
    leaderboard_rows: Iterable[dict[str, Any]],
    *,
    source_url: str,
    captured_at: str,
) -> dict[str, Any]:
    if not is_official_server_url(source_url):
        raise ValueError(
            "Leaderboard source must be https://benchmark.mvtec.com/."
        )
    try:
        datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Leaderboard captured_at must be ISO-8601.") from exc
    rows = list(leaderboard_rows)
    submission_id = str(evaluation["submission_id"])
    ours = next(
        (row for row in rows if row["submission_id"] == submission_id),
        None,
    )
    if ours is None:
        raise ValueError(
            "Official evaluation submission_id is absent from leaderboard."
        )
    official_means = evaluation["summary"]["means"]
    mismatches = {
        metric: {
            "official": float(official_means[metric]),
            "leaderboard": float(ours[metric]),
        }
        for metric in CORE_METRICS
        if not math.isclose(
            float(official_means[metric]),
            float(ours[metric]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    }
    if mismatches:
        raise ValueError(
            f"Leaderboard row does not match official evaluation: {mismatches}"
        )
    competitors = [
        row for row in rows if row["submission_id"] != submission_id
    ]
    metric_ranks = {}
    strict_best = {}
    for metric in CORE_METRICS:
        score = float(ours[metric])
        metric_ranks[metric] = 1 + sum(
            float(row[metric]) > score for row in competitors
        )
        strict_best[metric] = all(
            score > float(row[metric]) for row in competitors
        )
    supported = all(strict_best.values())
    return {
        "source_url": source_url,
        "captured_at": captured_at,
        "methods": len(rows),
        "submission_id": submission_id,
        "method": ours["method"],
        "metric_ranks": metric_ranks,
        "strict_best": strict_best,
        "sota_claim_supported": supported,
        "supported_claim": (
            "Top-ranked on all three official core metrics in this dated "
            "MVTec AD 2 leaderboard snapshot."
            if supported
            else (
                "Officially evaluated, but not strictly top-ranked on every "
                "core metric in this leaderboard snapshot."
            )
        ),
    }


def render_leaderboard_claim(result: dict[str, Any]) -> str:
    if result["sota_claim_supported"]:
        return (
            "On the dated MVTec AD 2 leaderboard snapshot captured at "
            f"`{result['captured_at']}`, Lite-SEER-AD ranks first and is "
            "strictly higher than every listed comparator on Image AUROC, "
            "Pixel AUROC, and AUPRO. This claim is limited to that snapshot "
            "and official benchmark protocol."
        )
    ranks = result["metric_ranks"]
    return (
        "Lite-SEER-AD has an official MVTec AD 2 evaluation, but the dated "
        "leaderboard snapshot does not support a strict all-core-metric SOTA "
        "claim. Ranks are Image AUROC "
        f"{ranks['image_auroc']}, Pixel AUROC {ranks['pixel_auroc']}, and "
        f"AUPRO {ranks['aupro']}."
    )


def write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "evaluation_scope",
        "evaluation_source",
        "seed",
        "categories",
        *CORE_METRICS,
        *OPTIONAL_METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_results_section(evaluation: dict[str, Any]) -> str:
    means = evaluation["summary"]["means"]
    lines = [
        "## MVTec AD 2 Official Evaluation",
        "",
        (
            "The seed-7 private submission passed the official local checker "
            "and was evaluated by the MVTec benchmark server."
        ),
        "",
        "| Image AUROC | Pixel AUROC | AUPRO |",
        "|---:|---:|---:|",
        (
            f"| {means['image_auroc']:.4f} | {means['pixel_auroc']:.4f} | "
            f"{means['aupro']:.4f} |"
        ),
        "",
        (
            f"Official submission ID: `{evaluation['submission_id']}`; "
            f"evaluated at `{evaluation['evaluated_at']}`."
        ),
        "",
        (
            "These scores establish official MVTec AD 2 evaluation. They do "
            "not by themselves establish a universal SOTA claim; that requires "
            "a dated, like-for-like leaderboard comparison."
        ),
    ]
    return "\n".join(lines)


def replace_marked_section(
    text: str,
    section: str,
    *,
    start_marker: str = "<!-- MVTEC_AD2_RESULTS_START -->",
    end_marker: str = "<!-- MVTEC_AD2_RESULTS_END -->",
) -> str:
    if start_marker not in text or end_marker not in text:
        raise ValueError("Results draft is missing MVTec AD 2 result markers.")
    before, remainder = text.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    return (
        before
        + start_marker
        + "\n"
        + section.rstrip()
        + "\n"
        + end_marker
        + after
    )
