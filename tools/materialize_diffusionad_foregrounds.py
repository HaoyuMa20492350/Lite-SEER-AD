from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gdown
import requests
from PIL import Image


MVTEC_FOREGROUND_FOLDER_ID = "1zHPkZWG8LsdGdKMacNoYE7aDhOeeiVVA"
MVTEC_FOREGROUND_URL = (
    "https://drive.google.com/drive/folders/"
    f"{MVTEC_FOREGROUND_FOLDER_ID}"
)
DRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm=t"
)
DRIVE_CATEGORY_ALIASES = {"metal_ nut": "metal_nut"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate the author-released DiffusionAD MVTec "
            "foreground masks, then materialize them beside MVTec AD."
        )
    )
    parser.add_argument(
        "--asset-root",
        default=(
            "third_party/official_baselines/diffusionad/"
            "foreground_assets/mvtec"
        ),
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--skip-materialize", action="store_true")
    return parser.parse_args()


def normalize_drive_path(path: str) -> Path:
    parts = list(Path(path.replace("\\", "/")).parts)
    if parts:
        parts[0] = DRIVE_CATEGORY_ALIASES.get(parts[0], parts[0])
    return Path(*parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def enumerate_drive_files(asset_root: Path) -> list[dict[str, str]]:
    files = gdown.download_folder(
        id=MVTEC_FOREGROUND_FOLDER_ID,
        output=str(asset_root),
        quiet=True,
        skip_download=True,
    )
    records = []
    for item in files:
        relative = normalize_drive_path(item.path)
        if relative.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        records.append(
            {
                "id": item.id,
                "drive_path": item.path.replace("\\", "/"),
                "relative_path": relative.as_posix(),
            }
        )
    return records


def download_one(
    record: dict[str, str],
    asset_root: Path,
    *,
    retries: int,
) -> dict[str, Any]:
    destination = asset_root / record["relative_path"]
    if valid_image(destination):
        return {
            **record,
            "status": "cached",
            "size_bytes": destination.stat().st_size,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = DRIVE_DOWNLOAD_URL.format(file_id=record["id"])
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            partial.write_bytes(response.content)
            if not valid_image(partial):
                raise ValueError("downloaded content is not a valid image")
            partial.replace(destination)
            return {
                **record,
                "status": "downloaded",
                "size_bytes": destination.stat().st_size,
            }
        except Exception as exc:
            last_error = str(exc)
            partial.unlink(missing_ok=True)
            time.sleep(min(8.0, 0.5 * 2**attempt))
    return {**record, "status": "failed", "error": last_error}


def download_assets(
    records: list[dict[str, str]],
    asset_root: Path,
    *,
    workers: int,
    retries: int,
) -> list[dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                download_one,
                record,
                asset_root,
                retries=retries,
            )
            for record in records
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"DiffusionAD foregrounds: {index}/{len(futures)}")
    return sorted(results, key=lambda item: item["relative_path"])


def materialize_dataset_assets(
    asset_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    records = []
    for source_category in sorted(
        path for path in asset_root.iterdir() if path.is_dir()
    ):
        category = DRIVE_CATEGORY_ALIASES.get(
            source_category.name,
            source_category.name,
        )
        for directory_name in ("DISthresh", "thresh"):
            source = source_category / directory_name
            if not source.is_dir():
                continue
            destination = dataset_root / category / directory_name
            shutil.copytree(source, destination, dirs_exist_ok=True)
            file_count = sum(1 for path in destination.rglob("*") if path.is_file())
            record = {
                "category": category,
                "directory": directory_name,
                "source": str(source),
                "destination": str(destination),
                "file_count": file_count,
            }
            if directory_name == "DISthresh":
                train_count = len(
                    list((dataset_root / category / "train" / "good").glob("*.png"))
                )
                record["train_good_count"] = train_count
                record["count_matches_train"] = file_count == train_count
            records.append(record)
    return {
        "dataset_root": str(dataset_root),
        "records": records,
        "complete": all(
            item.get("count_matches_train", True) for item in records
        ),
    }


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    records = enumerate_drive_files(asset_root)
    manifest_path = asset_root / "drive_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "folder_id": MVTEC_FOREGROUND_FOLDER_ID,
                "folder_url": MVTEC_FOREGROUND_URL,
                "file_count": len(records),
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results = download_assets(
        records,
        asset_root,
        workers=args.workers,
        retries=args.retries,
    )
    failures = [item for item in results if item["status"] == "failed"]
    completed = [item for item in results if item["status"] != "failed"]
    report: dict[str, Any] = {
        "folder_id": MVTEC_FOREGROUND_FOLDER_ID,
        "folder_url": MVTEC_FOREGROUND_URL,
        "asset_root": str(asset_root),
        "manifest_path": str(manifest_path),
        "expected_files": len(records),
        "completed_files": len(completed),
        "downloaded_files": sum(
            item["status"] == "downloaded" for item in completed
        ),
        "cached_files": sum(item["status"] == "cached" for item in completed),
        "total_bytes": sum(int(item["size_bytes"]) for item in completed),
        "failures": failures,
        "complete": not failures and len(completed) == len(records),
    }
    if report["complete"]:
        report["aggregate_sha256"] = hashlib.sha256(
            "\n".join(
                f"{item['relative_path']}:{sha256_file(asset_root / item['relative_path'])}"
                for item in completed
            ).encode("utf-8")
        ).hexdigest()
        if not args.skip_materialize:
            report["materialization"] = materialize_dataset_assets(
                asset_root,
                dataset_root,
            )
            report["complete"] = bool(
                report["materialization"]["complete"]
            )
    report_path = asset_root / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
