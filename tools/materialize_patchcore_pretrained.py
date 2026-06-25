from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest
from tools.audit_official_source_environments import parse_lfs_pointer


DEFAULT_BUNDLE = "IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3"
MVTEC15_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize pinned PatchCore Git LFS model files with hash checks."
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/patchcore",
    )
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--categories", default="all")
    parser.add_argument(
        "--report",
        default="tables/official_baseline_readiness/patchcore_materialization.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_categories(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MVTEC15_CATEGORIES)
    categories = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(categories) - set(MVTEC15_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown MVTec categories: {', '.join(unknown)}")
    return categories


def media_url(repository: str, commit: str, relative_path: Path) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError(f"Unsupported GitHub repository: {repository}")
    slug = repository[len(prefix) :].strip("/").removesuffix(".git")
    encoded = "/".join(
        urllib.parse.quote(part, safe="") for part in relative_path.parts
    )
    return (
        f"https://media.githubusercontent.com/media/{slug}/{commit}/{encoded}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_file(
    path: Path,
    *,
    source_root: Path,
    repository: str,
    commit: str,
    dry_run: bool,
) -> dict[str, Any]:
    pointer = parse_lfs_pointer(path)
    if pointer is None:
        return {
            "path": str(path),
            "status": "materialized",
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
    relative = path.relative_to(source_root)
    url = media_url(repository, commit, relative)
    record = {
        "path": str(path),
        "status": "planned" if dry_run else "downloaded",
        "size": pointer["size"],
        "sha256": pointer["sha256"],
        "url": url,
    }
    if dry_run:
        return record
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Lite-SEER-AD-PatchCore-LFS-materializer"},
    )
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.name}.",
        suffix=".download",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                shutil.copyfileobj(response, handle)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        actual_size = tmp_path.stat().st_size
        actual_hash = sha256(tmp_path)
        if actual_size != pointer["size"]:
            raise ValueError(
                f"{relative}: expected {pointer['size']} bytes, got {actual_size}"
            )
        if actual_hash != pointer["sha256"]:
            raise ValueError(
                f"{relative}: expected sha256 {pointer['sha256']}, got {actual_hash}"
            )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return record


def main() -> None:
    args = parse_args()
    manifest = load_official_source_manifest(args.manifest)
    source = manifest["sources"]["patchcore"]
    source_root = Path(args.source_root).resolve()
    categories = selected_categories(args.categories)
    bundle_root = source_root / "models" / args.bundle / "models"
    if not bundle_root.exists():
        raise FileNotFoundError(f"PatchCore model bundle is missing: {bundle_root}")
    records = []
    failures = []
    for category in categories:
        model_dir = bundle_root / f"mvtec_{category}"
        if not model_dir.exists():
            failures.append(
                {"category": category, "error": f"model directory missing: {model_dir}"}
            )
            continue
        for path in sorted(model_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                records.append(
                    {
                        "category": category,
                        **materialize_file(
                            path,
                            source_root=source_root,
                            repository=source["repository"],
                            commit=source["commit"],
                            dry_run=args.dry_run,
                        ),
                    }
                )
            except Exception as exc:
                failures.append(
                    {"category": category, "path": str(path), "error": str(exc)}
                )
    report = {
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "bundle": args.bundle,
        "categories": categories,
        "dry_run": bool(args.dry_run),
        "files": len(records),
        "downloaded": sum(row["status"] == "downloaded" for row in records),
        "materialized": sum(
            row["status"] in {"downloaded", "materialized"} for row in records
        ),
        "failures": failures,
        "complete": not failures
        and all(row["status"] != "planned" for row in records),
        "records": records,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
