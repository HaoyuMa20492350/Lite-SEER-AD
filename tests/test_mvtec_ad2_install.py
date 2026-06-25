from __future__ import annotations

import zipfile
import tarfile
import io
from pathlib import Path, PurePosixPath

import pytest

from seer_ad_v2.data.mvtec_ad2_discovery import (
    MVTEC_AD2_CATEGORIES,
    MVTEC_AD2_REQUIRED_PATHS,
)
from seer_ad_v2.data.mvtec_ad2_install import (
    install_archive,
    install_category_archives,
    relative_dataset_member,
    safe_target,
)


def build_complete_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for category in MVTEC_AD2_CATEGORIES:
            for required in MVTEC_AD2_REQUIRED_PATHS:
                archive.writestr(
                    f"release/MVTec-AD2/{category}/{required}/sample.png",
                    b"sample",
                )


def build_category_archives(path: Path) -> None:
    path.mkdir()
    for category in MVTEC_AD2_CATEGORIES:
        with tarfile.open(path / f"{category}.tar.gz", "w:gz") as archive:
            for required in MVTEC_AD2_REQUIRED_PATHS:
                payload = b"sample"
                info = tarfile.TarInfo(
                    f"release/MVTec-AD2/{category}/{required}/sample.png"
                )
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_install_complete_archive(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    build_complete_archive(archive)
    out = tmp_path / "installed"
    report = install_archive(archive, out)
    assert report["ready"] is True
    assert report["archive_root_prefix"] == "release/MVTec-AD2"
    assert report["matched_paths"] == report["required_paths"] == 56
    assert (
        out / "can" / "test_public" / "ground_truth" / "bad" / "sample.png"
    ).is_file()


def test_install_category_archives(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    build_category_archives(archives)
    out = tmp_path / "installed"
    report = install_category_archives(archives, out)
    assert report["ready"] is True
    assert report["matched_paths"] == report["required_paths"] == 56
    assert len(report["archives"]) == 8
    assert (
        out / "walnuts" / "test_private_mixed" / "sample.png"
    ).is_file()


def test_install_refuses_nonempty_target(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    build_complete_archive(archive)
    out = tmp_path / "installed"
    out.mkdir()
    (out / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to merge"):
        install_archive(archive, out)
    assert (out / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_archive_path_safety(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        relative_dataset_member("release/can/../../outside", "release")
    with pytest.raises(ValueError, match="escapes"):
        safe_target(tmp_path, PurePosixPath("can", "..", "..", "outside"))
