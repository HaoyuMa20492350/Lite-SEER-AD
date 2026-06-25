from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest


LFS_HEADER = "version https://git-lfs.github.com/spec/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cached official baseline sources and large-file readiness."
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines",
    )
    parser.add_argument(
        "--out",
        default="tables/official_baseline_readiness/source_environment_status",
    )
    return parser.parse_args()


def parse_lfs_pointer(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith(LFS_HEADER):
        return None
    values = {}
    for line in text.splitlines()[1:]:
        key, _, value = line.partition(" ")
        values[key] = value
    oid = values.get("oid", "")
    size = values.get("size", "")
    if not oid.startswith("sha256:") or not size.isdigit():
        return None
    return {"sha256": oid.removeprefix("sha256:"), "size": int(size)}


def _entrypoint_status(root: Path, entrypoint: str) -> tuple[str, bool]:
    token = entrypoint.split()[0]
    path_like = "/" in token or "\\" in token or Path(token).suffix != ""
    if path_like:
        path = root / token
        return str(path), path.exists()
    packaging_exists = any(
        (root / name).exists()
        for name in ("pyproject.toml", "setup.py", "setup.cfg")
    )
    return f"CLI:{token}", packaging_exists


def audit_source(
    method: str,
    source: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    root = source_root / method
    marker_path = root / ".lite_seer_source.json"
    marker = {}
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            marker = {}
    entrypoint, entrypoint_exists = _entrypoint_status(
        root,
        str(source["entrypoint"]),
    )
    pointers = []
    if root.exists():
        for path in root.rglob("*"):
            pointer = parse_lfs_pointer(path)
            if pointer is not None:
                pointers.append((path, pointer))
    requirements = [
        str(path.relative_to(root))
        for name in ("requirements.txt", "pyproject.toml", "setup.py", "uv.lock")
        if (path := root / name).exists()
    ]
    errors = []
    if not root.exists():
        errors.append("source directory missing")
    if marker.get("commit") != source["commit"]:
        errors.append("source marker missing or commit mismatch")
    if not entrypoint_exists:
        errors.append("declared entrypoint missing")
    return {
        "method": method,
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "source_exists": root.exists(),
        "marker_matches": marker.get("commit") == source["commit"],
        "entrypoint": entrypoint,
        "entrypoint_exists": entrypoint_exists,
        "requirements": ";".join(requirements),
        "lfs_pointer_files": len(pointers),
        "lfs_pointer_bytes": sum(item["size"] for _, item in pointers),
        "pretrained_assets_ready": not pointers,
        "source_ready": not errors,
        "errors": "; ".join(errors),
        "asset_notes": (
            f"{len(pointers)} Git LFS objects remain pointers"
            if pointers
            else ""
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = load_official_source_manifest(args.manifest)
    source_root = Path(args.source_root)
    rows = [
        audit_source(method, source, source_root)
        for method, source in sorted(manifest["sources"].items())
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "table_source_environment_status.csv", rows)
    summary = {
        "manifest": str(Path(args.manifest)),
        "source_root": str(source_root),
        "methods": len(rows),
        "source_ready": sum(bool(row["source_ready"]) for row in rows),
        "pretrained_assets_ready": sum(
            bool(row["pretrained_assets_ready"]) for row in rows
        ),
        "lfs_pointer_files": sum(int(row["lfs_pointer_files"]) for row in rows),
        "lfs_pointer_bytes": sum(int(row["lfs_pointer_bytes"]) for row in rows),
        "complete": all(bool(row["source_ready"]) for row in rows),
        "records": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
