from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from seer_ad_v2.data.mvtec_ad2_discovery import (
    MVTEC_AD2_CATEGORIES,
    analyse_member_names,
    archive_member_names,
    inspect_dataset_directory,
)


def ready_root_prefix(names: Iterable[str]) -> str:
    candidates = analyse_member_names(names)
    ready = [row for row in candidates if row["ready"]]
    if not ready:
        raise ValueError("Archive does not contain a complete MVTec AD 2 layout.")
    return str(ready[0]["root_prefix"])


def category_root_prefix(
    names: Iterable[str],
    expected_category: str,
) -> str:
    if expected_category not in MVTEC_AD2_CATEGORIES:
        raise ValueError(f"Unknown MVTec AD 2 category: {expected_category}")
    candidates = analyse_member_names(names)
    ready = [
        row
        for row in candidates
        if expected_category not in row["missing"]
    ]
    if not ready:
        raise ValueError(
            f"Archive does not contain a complete {expected_category} layout."
        )
    return str(ready[0]["root_prefix"])


def relative_dataset_member(name: str, root_prefix: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized:
        return None
    if root_prefix:
        prefix = f"{root_prefix.strip('/')}/"
        if not normalized.startswith(prefix):
            return None
        normalized = normalized[len(prefix) :]
    relative = PurePosixPath(normalized)
    if not relative.parts or relative.parts[0] not in MVTEC_AD2_CATEGORIES:
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return relative


def safe_target(root: Path, relative: PurePosixPath) -> Path:
    root = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes target root: {relative}") from exc
    return target


def _ensure_empty_target(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Target must be absent or empty; refusing to merge into {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _extract_zip(path: Path, out: Path, root_prefix: str) -> int:
    files = 0
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            relative = relative_dataset_member(item.filename, root_prefix)
            if relative is None or item.is_dir():
                continue
            target = safe_target(out, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            files += 1
    return files


def _extract_tar(
    path: Path,
    out: Path,
    root_prefix: str,
    expected_category: str | None = None,
) -> int:
    files = 0
    with tarfile.open(path, "r:*") as archive:
        for item in archive:
            relative = relative_dataset_member(item.name, root_prefix)
            if relative is None or item.isdir():
                continue
            if (
                expected_category is not None
                and relative.parts[0] != expected_category
            ):
                raise ValueError(
                    "Category archive contains an unexpected category: "
                    f"{relative.parts[0]}"
                )
            if not item.isfile():
                raise ValueError(
                    f"Refusing non-regular archive member: {item.name}"
                )
            source = archive.extractfile(item)
            if source is None:
                raise ValueError(f"Unable to read archive member: {item.name}")
            target = safe_target(out, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            files += 1
    return files


def install_archive(path: Path, out: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    names = list(archive_member_names(path))
    root_prefix = ready_root_prefix(names)
    _ensure_empty_target(out)
    if zipfile.is_zipfile(path):
        files = _extract_zip(path, out, root_prefix)
    elif tarfile.is_tarfile(path):
        files = _extract_tar(path, out, root_prefix)
    else:
        raise ValueError(f"Unsupported archive: {path}")
    readiness = inspect_dataset_directory(out)
    if not readiness["ready"]:
        raise RuntimeError(
            "Extracted archive failed MVTec AD 2 readiness validation."
        )
    return {
        "archive": str(path),
        "dataset_root": str(out.resolve()),
        "archive_root_prefix": root_prefix,
        "files_extracted": files,
        "ready": True,
        "matched_paths": readiness["matched_paths"],
        "required_paths": readiness["required_paths"],
    }


def install_category_archives(
    directory: Path,
    out: Path,
) -> dict[str, Any]:
    directory = directory.resolve()
    archives = {
        category: directory / f"{category}.tar.gz"
        for category in MVTEC_AD2_CATEGORIES
    }
    missing = [
        str(path)
        for path in archives.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing MVTec AD 2 category archives: " + ", ".join(missing)
        )
    _ensure_empty_target(out)
    rows = []
    total_files = 0
    for category, path in archives.items():
        names = list(archive_member_names(path))
        root_prefix = category_root_prefix(names, category)
        if zipfile.is_zipfile(path):
            raise ValueError(
                "Per-category installer currently expects official TAR archives."
            )
        if not tarfile.is_tarfile(path):
            raise ValueError(f"Unsupported archive: {path}")
        files = _extract_tar(
            path,
            out,
            root_prefix,
            expected_category=category,
        )
        total_files += files
        rows.append(
            {
                "category": category,
                "archive": str(path),
                "archive_root_prefix": root_prefix,
                "files_extracted": files,
            }
        )
    readiness = inspect_dataset_directory(out)
    if not readiness["ready"]:
        raise RuntimeError(
            "Extracted category archives failed MVTec AD 2 readiness validation."
        )
    return {
        "archive_directory": str(directory),
        "dataset_root": str(out.resolve()),
        "archives": rows,
        "files_extracted": total_files,
        "ready": True,
        "matched_paths": readiness["matched_paths"],
        "required_paths": readiness["required_paths"],
    }
