from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import gdown


DDAD_FOLDER_ID = "1FF83llo3a-mN5pJN8-_mw0hL5eZqe9fC"
DDAD_MVTEC_FILES = {
    "bottle/1000": "1qZjfuOM5YFKWtUSY9wkDtDWp8E5EssuP",
    "bottle/feat8": "1EQNhJhurrPtGFj7g8CYBLXdGC7rhgduV",
    "cable/3000": "18TVa1eEWd6tHUtneWW6URYsrQy16Zmim",
    "cable/feat0": "1gr1jzjcGi3Nmoanw7XNKaaisBmY8gh9c",
    "capsule/1500": "1HeGXMknT0cDmdwiDnb7X44D-lYuqOnVW",
    "capsule/feat8": "18DScR0RrtoAbfuWnsd0ZMEtZRACdnxtH",
    "carpet/2500": "1J_HYd1ilyYZ3-klGlII6ETrbjz-IER6F",
    "carpet/feat0": "130kiheoa871zF6l1V4T6_mtW9QrakC1y",
    "grid/2000": "18CsJB-TAlNRElczYIm1dmYZ67Coh1C4w",
    "grid/feat6": "115v1UMZOivyXtS74p4TVs6RBjFIUHPZW",
    "hazelnut/2000": "1hQhMU4VvGq3IPQDCmildYSINOsarxrSm",
    "hazelnut/feat3": "14pXSaDrSRcUSd8UfxVpecFTmYo0aunG1",
    "leather/2000": "19_LHLQaQVxI9bVk5ZcU815m7n2eyulzc",
    "leather/feat5": "1YRgpieD6G9eAv9IyZ07MGctwCHMF33H0",
    "metal_nut/3000": "1Mh9IgMl5u7sOLaDpzDMH4C1lnDYxW7uU",
    "metal_nut/feat1": "1oUzFGcgQ0Ka7GIA4ntlMhRPNNptYjOth",
    "pill/1000": "1EUfzyG5EMOts2JRMX6n0EgtD_FRAP8dk",
    "pill/feat4": "1SK5Jm0H25b56ucSh41P-z_BljeFY_udf",
    "screw/2000": "1eygWvm_z7gW6j3TGui41DNB5f0zVQWEZ",
    "screw/feat4": "1OYLMjrE56gDMgBzJAKD4Yoj2GBpgmssd",
    "tile/1000": "1mh2tcrpdBjCLRe7rVs4vvLxLgGEYka4b",
    "tile/feat0": "1EZG5OGx2IuNEzjff3iLeeHrnRyQYwBJO",
    "toothbrush/2000": "1JbR2SwRU25xTe4Rswdsf4GwAAC5LeNjI",
    "toothbrush/feat2": "1FFxzlJsCHJq261d9pAfQGlWFosrrTtc_",
    "transistor/2000": "1uTTeE6cZs64jgX_XSVsx0-jJ3ESi2jgI",
    "transistor/feat0": "1X-edaUr_Uso5pcv1fpnFGeUBARMAl1FZ",
    "wood/2000": "1YA5uG37aQJ7maYC5TwGmwfhrEGGYUHo4",
    "wood/feat16": "1kk3MAgoIC-MchXTFAQF_RMO-BHjJrKDO",
    "zipper/1000": "1GH1yB6N1-T1-hj3uieWDMclaNZPvi8ng",
    "zipper/feat6": "1IcCLDaP5tC1QLqaBQ4Y796Ekuc_NY7yj",
}
DDAD_MVTEC_SETTINGS = {
    "bottle": {"unet_checkpoint": "1000", "feature_checkpoint": "feat8", "w": 3},
    "cable": {"unet_checkpoint": "3000", "feature_checkpoint": "feat0", "w": 3},
    "capsule": {"unet_checkpoint": "1500", "feature_checkpoint": "feat8", "w": 8},
    "carpet": {"unet_checkpoint": "2500", "feature_checkpoint": "feat0", "w": 0},
    "grid": {"unet_checkpoint": "2000", "feature_checkpoint": "feat6", "w": 4},
    "hazelnut": {"unet_checkpoint": "2000", "feature_checkpoint": "feat3", "w": 5},
    "leather": {"unet_checkpoint": "2000", "feature_checkpoint": "feat5", "w": 11},
    "metal_nut": {"unet_checkpoint": "3000", "feature_checkpoint": "feat1", "w": 7},
    "pill": {"unet_checkpoint": "1000", "feature_checkpoint": "feat4", "w": 9},
    "screw": {"unet_checkpoint": "2000", "feature_checkpoint": "feat4", "w": 2},
    "tile": {"unet_checkpoint": "1000", "feature_checkpoint": "feat0", "w": 4},
    "toothbrush": {"unet_checkpoint": "2000", "feature_checkpoint": "feat2", "w": 0},
    "transistor": {"unet_checkpoint": "2000", "feature_checkpoint": "feat0", "w": 0},
    "wood": {"unet_checkpoint": "2000", "feature_checkpoint": "feat16", "w": 11},
    "zipper": {"unet_checkpoint": "1000", "feature_checkpoint": "feat6", "w": 10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the author-released DDAD MVTec checkpoints."
    )
    parser.add_argument(
        "--out",
        default="third_party/official_baselines/ddad/pretrained/MVTec",
    )
    parser.add_argument("--categories", default="all")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_categories(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(DDAD_MVTEC_SETTINGS)
    categories = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]
    unknown = sorted(set(categories) - set(DDAD_MVTEC_SETTINGS))
    if unknown:
        raise ValueError("Unknown DDAD categories: " + ", ".join(unknown))
    return categories


def download_with_retry(
    file_id: str,
    path: Path,
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 2.0,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = gdown.download(
                id=file_id,
                output=str(path),
                quiet=False,
                resume=True,
            )
            if result is None or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"gdown did not materialize a non-empty file: {path}")
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = initial_delay_seconds * (2 ** (attempt - 1))
            print(
                f"Download attempt {attempt}/{attempts} failed for {path}: "
                f"{exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Failed to download {path} after {attempts} attempts"
    ) from last_error


def materialize(
    out_root: Path,
    categories: list[str],
) -> dict[str, Any]:
    records = []
    for category in categories:
        settings = DDAD_MVTEC_SETTINGS[category]
        for filename in (
            settings["unet_checkpoint"],
            settings["feature_checkpoint"],
        ):
            relative = f"{category}/{filename}"
            file_id = DDAD_MVTEC_FILES[relative]
            path = out_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            status = "cached"
            if not path.exists():
                download_with_retry(file_id, path)
                status = "downloaded"
            records.append(
                {
                    "category": category,
                    "filename": filename,
                    "google_drive_id": file_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "status": status,
                }
            )
    report = {
        "source_folder_id": DDAD_FOLDER_ID,
        "categories": categories,
        "files": len(records),
        "complete": len(records) == 2 * len(categories),
        "records": records,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    report = materialize(
        Path(args.out),
        selected_categories(args.categories),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
