"""Build a SHA256 manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PATTERNS = (
    "README.md",
    "REPRODUCE.md",
    "MODEL_CARD.md",
    "DATASETS.md",
    "CITATION.cff",
    ".zenodo.json",
    "release_links.template.json",
    "release_metadata.template.json",
    "release_metadata.json",
    "submission_metadata.template.json",
    "submission_metadata.json",
    "second_hardware_profile.json",
    "second_hardware_energy.json",
    "paper/manuscript.md",
    "paper/supplement.md",
    "paper/references.bib",
    "*.py",
    "docs/*.md",
    "configs/**/*.yaml",
    "tools/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
    "tables/strict_fixed_threshold/**/*.csv",
    "tables/strict_fixed_threshold/**/*.json",
    "tables/feature_first_fusion_aggregate_paper_package/**/*.csv",
    "tables/feature_first_fusion_aggregate_paper_package/**/*.json",
    "tables/feature_first_fusion_aggregate_paper_package/**/*.md",
    "tables/external_baseline_comparison/**/*.csv",
    "tables/external_baseline_comparison/**/*.json",
    "tables/diffusionad_compute_plan/**/*.json",
    "tables/diffusionad_compute_plan/**/*.csv",
    "tables/image_score_aggregation_mvtec15/**/*.json",
    "tables/image_score_aggregation_mvtec15/**/*.csv",
    "tables/failure_taxonomy/**/*.csv",
    "tables/failure_taxonomy/**/*.json",
    "tables/failure_taxonomy/**/*.md",
    "tables/final_external_handoff/**/*.csv",
    "tables/final_external_handoff/**/*.json",
    "tables/final_external_handoff/**/*.md",
    "tables/final_external_handoff/**/*.ps1",
    "tables/completion_gap_matrix/**/*.csv",
    "tables/completion_gap_matrix/**/*.json",
    "tables/completion_gap_matrix/**/*.md",
    "tables/claim_traceability/**/*.csv",
    "tables/claim_traceability/**/*.json",
    "tables/claim_traceability/**/*.md",
    "tables/release_readiness/**/*.csv",
    "tables/release_readiness/**/*.json",
    "tables/release_readiness/**/*.md",
    "tables/lc_rds_budget_audit/**/*.csv",
    "tables/lc_rds_budget_audit/**/*.json",
    "tables/hn_sev_retention_calibration/**/*.csv",
    "tables/hn_sev_retention_calibration/**/*.json",
    "tables/hn_sev_retention_calibration/**/*.png",
    "tables/hn_sev_input_ablation/**/*.csv",
    "tables/hn_sev_input_ablation/**/*.json",
    "tables/repair_executor_ablation/**/*.csv",
    "tables/repair_executor_ablation/**/*.json",
    "tables/repair_executor_ablation/**/*.png",
    "tables/deployment_latency/**/*.csv",
    "tables/deployment_latency/**/*.json",
    "tables/deployment_production_latency/**/*.csv",
    "tables/deployment_production_latency/**/*.json",
    "tables/deployment_production_latency/**/*.md",
    "tables/deployment_production_latency/**/*.ps1",
    "tables/deployment_readiness/**/*.csv",
    "tables/deployment_readiness/**/*.json",
    "tables/submission_package_readiness/**/*.csv",
    "tables/submission_package_readiness/**/*.json",
    "tables/submission_package_readiness/**/*.md",
    "tables/submission_package_readiness/**/*.ps1",
    "artifacts/thresholds/**/*.json",
    "artifacts/predictions_manifest.json",
    "artifacts/stats/**/*.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                files.add(path.resolve())
    return sorted(files)


def build_manifest(root: Path, patterns: list[str]) -> dict[str, object]:
    root = root.resolve()
    entries = []
    for path in collect_files(root, patterns):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "lite-seer-ad-artifact-manifest-v1",
        "root": root.as_posix(),
        "patterns": patterns,
        "file_count": len(entries),
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--pattern", action="append", dest="patterns")
    args = parser.parse_args()

    patterns = args.patterns or list(DEFAULT_PATTERNS)
    manifest = build_manifest(args.root, patterns)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote {manifest['file_count']} entries to {args.out}")


if __name__ == "__main__":
    main()
