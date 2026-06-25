from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


OFFICIAL_SUBMISSION_DIRS = {
    "anomaly_images",
    "anomaly_images_thresholded",
}


def default_metadata_dir(submission_dir: Path) -> Path:
    return submission_dir.with_name(f"{submission_dir.name}_metadata")


def assert_path_outside_submission(
    submission_dir: Path,
    artifact_path: Path,
    artifact_name: str,
) -> None:
    submission_resolved = submission_dir.resolve()
    artifact_resolved = artifact_path.resolve()
    if artifact_resolved == submission_resolved or artifact_resolved.is_relative_to(
        submission_resolved
    ):
        raise ValueError(
            f"{artifact_name} must be outside the official submission root: "
            f"{artifact_path}"
        )


def assert_submission_root(
    submission_dir: Path,
    *,
    require_thresholded: bool = True,
) -> None:
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"Submission directory does not exist: {submission_dir}")
    found = {path.name for path in submission_dir.iterdir()}
    required = {"anomaly_images"}
    if require_thresholded:
        required.add("anomaly_images_thresholded")
    unexpected = found - OFFICIAL_SUBMISSION_DIRS
    missing = required - found
    if unexpected or missing:
        details = []
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        if missing:
            details.append(f"missing={sorted(missing)}")
        raise ValueError(
            "MVTec AD 2 submission root is not checker-compatible: "
            + "; ".join(details)
        )
    non_dirs = [
        name for name in found if not (submission_dir / name).is_dir()
    ]
    if non_dirs:
        raise ValueError(
            f"MVTec AD 2 submission entries must be directories: {sorted(non_dirs)}"
        )


def assert_export_root_available(submission_dir: Path) -> None:
    if not submission_dir.exists():
        return
    unexpected = {
        path.name
        for path in submission_dir.iterdir()
        if path.name not in OFFICIAL_SUBMISSION_DIRS
    }
    if unexpected:
        raise ValueError(
            "Refusing to export into a submission root containing files that "
            f"the official checker rejects: {sorted(unexpected)}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_official_checker(
    submission_dir: Path,
    checker_path: Path,
) -> dict[str, Any]:
    checker_path = checker_path.resolve()
    submission_dir = submission_dir.resolve()
    if not checker_path.is_file():
        raise FileNotFoundError(f"Official checker not found: {checker_path}")
    code = (
        "import sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from check_and_prepare_data_for_upload import check_submission; "
        "check_submission(sys.argv[2])"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(checker_path.parent),
            str(submission_dir),
        ],
        cwd=checker_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            f"Official MVTec AD 2 checker failed for {submission_dir}:\n{output}"
        )
    return {
        "status": "passed",
        "checker": str(checker_path),
        "checker_sha256": sha256_file(checker_path),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def create_submission_archive(
    submission_dir: Path,
    archive_path: Path,
) -> Path:
    assert_submission_root(submission_dir)
    assert_path_outside_submission(
        submission_dir,
        archive_path,
        "Submission archive",
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.add(
                    path,
                    arcname=Path(
                        submission_dir.name,
                        path.relative_to(submission_dir),
                    ),
                )
    return archive_path
