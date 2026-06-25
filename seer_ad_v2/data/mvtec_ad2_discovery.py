from __future__ import annotations

import os
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


MVTEC_AD2_CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)

MVTEC_AD2_REQUIRED_PATHS = (
    "train/good",
    "validation/good",
    "test_public/good",
    "test_public/bad",
    "test_public/ground_truth/bad",
    "test_private",
    "test_private_mixed",
)

ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
)

PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "runs",
    "tables",
}


def _normalise_member_name(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def _path_is_under(relative: str, required: str) -> bool:
    return relative == required or relative.startswith(f"{required}/")


def analyse_member_names(names: Iterable[str]) -> list[dict[str, Any]]:
    coverage: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for raw_name in names:
        name = _normalise_member_name(raw_name)
        if not name:
            continue
        parts = name.split("/")
        for index, part in enumerate(parts):
            if part not in MVTEC_AD2_CATEGORIES:
                continue
            root_prefix = "/".join(parts[:index])
            relative = "/".join(parts[index + 1 :])
            for required in MVTEC_AD2_REQUIRED_PATHS:
                if _path_is_under(relative, required):
                    coverage[root_prefix][part].add(required)
            break

    candidates = []
    required = set(MVTEC_AD2_REQUIRED_PATHS)
    for root_prefix, categories in coverage.items():
        matched = sum(
            len(categories.get(category, set()))
            for category in MVTEC_AD2_CATEGORIES
        )
        missing = {
            category: sorted(required - categories.get(category, set()))
            for category in MVTEC_AD2_CATEGORIES
        }
        missing = {
            category: values
            for category, values in missing.items()
            if values
        }
        candidates.append(
            {
                "root_prefix": root_prefix,
                "ready": not missing,
                "matched_paths": matched,
                "required_paths": (
                    len(MVTEC_AD2_CATEGORIES)
                    * len(MVTEC_AD2_REQUIRED_PATHS)
                ),
                "missing": missing,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (-int(row["matched_paths"]), row["root_prefix"]),
    )


def inspect_dataset_directory(path: Path) -> dict[str, Any]:
    names = []
    for category in MVTEC_AD2_CATEGORIES:
        for required in MVTEC_AD2_REQUIRED_PATHS:
            candidate = path / category / Path(required)
            if candidate.is_dir():
                names.append(f"{category}/{required}/present")
    candidates = analyse_member_names(names)
    candidate = candidates[0] if candidates else {
        "root_prefix": "",
        "ready": False,
        "matched_paths": 0,
        "required_paths": (
            len(MVTEC_AD2_CATEGORIES) * len(MVTEC_AD2_REQUIRED_PATHS)
        ),
        "missing": {
            category: list(MVTEC_AD2_REQUIRED_PATHS)
            for category in MVTEC_AD2_CATEGORIES
        },
    }
    return {
        "source_type": "directory",
        "path": str(path.resolve()),
        **candidate,
    }


def archive_member_names(path: Path) -> Iterator[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                yield item.filename
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for item in archive:
                yield item.name
        return
    raise ValueError(f"Unsupported or invalid archive: {path}")


def inspect_archive(path: Path) -> dict[str, Any]:
    try:
        candidates = analyse_member_names(archive_member_names(path))
    except (
        EOFError,
        OSError,
        RuntimeError,
        tarfile.TarError,
        zipfile.BadZipFile,
        ValueError,
    ) as exc:
        return {
            "source_type": "archive",
            "path": str(path.resolve()),
            "ready": False,
            "error": str(exc),
            "candidates": [],
        }
    return {
        "source_type": "archive",
        "path": str(path.resolve()),
        "ready": any(row["ready"] for row in candidates),
        "candidates": candidates[:8],
    }


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def walk_candidates(root: Path, max_depth: int) -> Iterator[Path]:
    root = root.resolve()
    if not root.exists():
        return
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if set(MVTEC_AD2_CATEGORIES).issubset(directories):
            yield current_path
        directories[:] = [
            name
            for name in directories
            if name not in PRUNED_DIRECTORY_NAMES
            and (depth < max_depth)
        ]
        for name in files:
            path = current_path / name
            if _is_archive(path):
                yield path


def discover(roots: Iterable[Path], max_depth: int = 6) -> dict[str, Any]:
    seen: set[Path] = set()
    results = []
    scanned_archives = 0
    scanned_directories = 0
    for root in roots:
        for candidate in walk_candidates(root, max_depth):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            result = (
                inspect_archive(resolved)
                if resolved.is_file()
                else inspect_dataset_directory(resolved)
            )
            if resolved.is_file():
                scanned_archives += 1
            else:
                scanned_directories += 1
            if (
                result.get("ready")
                or result.get("candidates")
                or result.get("error")
            ):
                results.append(result)
    return {
        "official_categories": list(MVTEC_AD2_CATEGORIES),
        "required_paths_per_category": list(MVTEC_AD2_REQUIRED_PATHS),
        "roots": [str(Path(root).resolve()) for root in roots],
        "ready": any(row.get("ready") for row in results),
        "scanned_archives": scanned_archives,
        "scanned_dataset_directories": scanned_directories,
        "results": results,
    }
