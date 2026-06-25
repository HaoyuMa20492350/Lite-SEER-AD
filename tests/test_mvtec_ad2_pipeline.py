from __future__ import annotations

import json
from pathlib import Path

from seer_ad_v2.evaluation.mvtec_ad2_pipeline import (
    OFFICIAL_CATEGORIES,
    audit_pipeline,
    checkpoint_template,
    expected_public_runs,
    private_command,
    public_command,
)


def test_expected_runs_and_commands() -> None:
    runs = expected_public_runs()
    assert len(runs) == 24
    assert len({(row["category"], row["seed"]) for row in runs}) == 24
    assert checkpoint_template().format(category="can") == (
        "runs/mvtec_ad2_feature_first_can_seed7_models/feature_prior.pt"
    )
    public = public_command(
        "python",
        config="config.yaml",
        run_prefix="run",
        out="tables/ad2",
        log_file="tables/ad2/log.txt",
        device="cuda",
    )
    assert "--resume" in public
    assert "7,13,23" in public
    private = private_command(
        "python",
        config="config.yaml",
        run_prefix="run",
        private_seed=13,
        out="submission",
        metadata_out="metadata",
        archive_out="submission.tar.gz",
        device="cuda",
    )
    assert "runs/run_{category}_seed13_models/feature_prior.pt" in private
    assert "--archive-out" in private
    assert private[private.index("--submission-resolution") + 1] == "model"


def test_audit_complete_synthetic_pipeline(tmp_path: Path) -> None:
    for category in OFFICIAL_CATEGORIES:
        for required in (
            "train/good",
            "validation/good",
            "test_public/good",
            "test_public/bad",
            "test_public/ground_truth/bad",
            "test_private",
            "test_private_mixed",
        ):
            (
                tmp_path
                / "SEER-AD-dataset"
                / "MVTec-AD2"
                / category
                / Path(required)
            ).mkdir(parents=True)
    checker = (
        tmp_path
        / "official_mvtec_ad2_utils"
        / "MVTecAD2_public_code_utils"
        / "check_and_prepare_data_for_upload.py"
    )
    checker.parent.mkdir(parents=True)
    checker.write_text("# checker", encoding="utf-8")

    table_dir = tmp_path / "tables" / "mvtec_ad2_feature_first"
    table_dir.mkdir(parents=True)
    rows = expected_public_runs()
    headers = "dataset,category,seed,run\n"
    lines = [
        f"mvtec_ad2,{row['category']},{row['seed']},{row['public_run']}"
        for row in rows
    ]
    (table_dir / "table_public_runs.csv").write_text(
        headers + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (table_dir / "protocol.json").write_text(
        json.dumps(
            {
                "runs_completed": 24,
                "categories": list(OFFICIAL_CATEGORIES),
                "seeds": [7, 13, 23],
            }
        ),
        encoding="utf-8",
    )
    for row in rows:
        model = tmp_path / "runs" / row["model_run"]
        public = tmp_path / "runs" / row["public_run"]
        model.mkdir(parents=True)
        public.mkdir(parents=True)
        (model / "feature_prior.pt").write_bytes(b"checkpoint")
        (public / "predictions.npz").write_bytes(b"predictions")
        (public / "metrics.json").write_text("{}", encoding="utf-8")

    submission = tmp_path / "submissions" / "mvtec_ad2_seed7_model256"
    submission.mkdir(parents=True)
    metadata = (
        tmp_path
        / "submissions"
        / "mvtec_ad2_seed7_model256_metadata"
    )
    metadata.mkdir(parents=True)
    (metadata / "submission_protocol.json").write_text(
        json.dumps(
            {
                "files": 4090,
                "full_official_submission": True,
                "official_checker": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    (
        tmp_path
        / "submissions"
        / "mvtec_ad2_seed7_model256.tar.gz"
    ).write_bytes(b"archive")
    report = audit_pipeline(tmp_path)
    assert report["dataset_ready"] is True
    assert report["public_complete"] is True
    assert report["private_complete"] is True
    assert report["pipeline_complete"] is True
