from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.export_deployment_latency_audit import (
    COMPONENTS,
    benchmark_components,
    write_outputs,
)


def make_sample_repo(root: Path) -> None:
    image_path = root / "SEER-AD-dataset/MVTec-AD/bottle/train/good/000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), (80, 90, 100)).save(image_path)
    summary_path = root / "tables/lc_rds_budget_audit/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"max_budget_violation_rate": 0.0}),
        encoding="utf-8",
    )


def test_benchmark_components_records_all_required_components(tmp_path: Path) -> None:
    make_sample_repo(tmp_path)

    rows, info = benchmark_components(tmp_path, device="cpu", warmups=1, repeats=2)

    assert [row["component"] for row in rows] == COMPONENTS
    assert info["device"] == "cpu"
    for row in rows:
        assert row["latency_protocol"] == "synchronized_batch_latency_v1"
        assert row["latency_batch_size"] == 1
        assert "latency_ms_p95" in row
        assert "latency_ms_p99" in row


def test_write_outputs_keeps_release_gate_false_for_smoke_audit(tmp_path: Path) -> None:
    make_sample_repo(tmp_path)
    out_dir = tmp_path / "tables/deployment_latency"

    summary = write_outputs(
        tmp_path,
        out_dir,
        device="cpu",
        warmups=1,
        repeats=2,
    )

    assert summary["evidence_level"] == "synchronized_component_smoke_v1"
    assert summary["release_gate_passed"] is False
    assert summary["latency_ms_p99"] >= summary["latency_ms_p95"]
    assert summary["budget_violation_rate"] == 0.0
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_component_latency.csv").is_file()
