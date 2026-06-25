from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch pinned official baseline source archives from GitHub."
    )
    parser.add_argument(
        "--manifest",
        default="baselines/official_sources.json",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods or all.",
    )
    parser.add_argument(
        "--out",
        default="third_party/official_baselines",
    )
    parser.add_argument(
        "--report",
        default="tables/official_baseline_readiness/source_fetch_status.json",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _selected(value: str, available: set[str]) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(available)
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(methods) - available)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _within_workspace(path: Path) -> Path:
    resolved = path.resolve()
    workspace = REPO_ROOT.resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError(f"Path is outside workspace: {resolved}")
    return resolved


def _repo_slug(repository: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError(f"Unsupported repository URL: {repository}")
    return repository[len(prefix) :].strip("/").removesuffix(".git")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination = _within_workspace(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name and not name.endswith("/")]
        if not names:
            raise ValueError(f"{archive} is empty")
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1:
            raise ValueError(f"{archive} has multiple archive roots: {sorted(roots)}")
        root_name = next(iter(roots))
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(
                    f"Archive path escapes destination: {member.filename}"
                )
        bundle.extractall(destination)
    return destination / root_name


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Lite-SEER-AD-official-baseline-fetcher",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def fetch_source(
    method: str,
    source: dict[str, Any],
    out_root: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    method_dir = _within_workspace(out_root / method)
    marker = method_dir / ".lite_seer_source.json"
    if marker.exists() and not refresh:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("commit") == source["commit"]:
            return {**existing, "status": "cached"}

    if method_dir.exists():
        _within_workspace(method_dir)
        shutil.rmtree(method_dir)
    method_dir.parent.mkdir(parents=True, exist_ok=True)

    slug = _repo_slug(str(source["repository"]))
    archive_url = (
        f"https://api.github.com/repos/{slug}/zipball/{source['commit']}"
    )
    with tempfile.TemporaryDirectory(
        prefix=f"lite-seer-{method}-",
        dir=str(_within_workspace(out_root.parent)),
    ) as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / f"{method}.zip"
        extracted = tmp_dir / "extracted"
        _download(archive_url, archive)
        archive_sha256 = _sha256(archive)
        extracted_root = _safe_extract(archive, extracted)
        shutil.move(str(extracted_root), str(method_dir))

    record = {
        "method": method,
        "status": "fetched",
        "source_kind": source["source_kind"],
        "repository": source["repository"],
        "commit": source["commit"],
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "destination": str(method_dir),
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    manifest = load_official_source_manifest(args.manifest)
    sources = manifest["sources"]
    methods = _selected(args.methods, set(sources))
    out_root = _within_workspace(Path(args.out))
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for method in methods:
        try:
            records.append(
                fetch_source(
                    method,
                    sources[method],
                    out_root,
                    refresh=args.refresh,
                )
            )
        except Exception as exc:
            failures.append({"method": method, "error": str(exc)})
    report = {
        "manifest": str(Path(args.manifest)),
        "out": str(out_root),
        "requested_methods": methods,
        "fetched_or_cached": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(methods),
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
