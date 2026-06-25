from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.export_repair_executor_ablation import (
    DIFFUSION_EXECUTOR_NAME,
    DiffusionCheckpointRegistry,
    build_ablation,
    collect_normal_images,
    synthetic_defect,
    write_outputs,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:128, :128]
    for channel, value in enumerate(color):
        base[..., channel] = np.clip(value + (xx + yy + channel * 17) % 23, 0, 255)
    Image.fromarray(base, mode="RGB").save(path)


def make_dataset(root: Path) -> None:
    write_image(
        root / "SEER-AD-dataset/MVTec-AD/bottle/train/good/000.png",
        (80, 90, 100),
    )
    write_image(
        root / "SEER-AD-dataset/VisA/candle/Data/Images/Normal/0000.JPG",
        (110, 90, 70),
    )
    write_image(
        root / "SEER-AD-dataset/MPDD/official/MPDD/MPDD/bracket_black/train/good/000.png",
        (70, 110, 90),
    )


def test_collect_normal_images_finds_supported_dataset_layouts(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    rows = collect_normal_images(tmp_path, images_per_category=1)

    assert len(rows) == 3
    assert {row["dataset"] for row in rows} == {"mvtec15", "visa", "mpdd"}


def test_synthetic_defect_is_deterministic(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    path = tmp_path / "SEER-AD-dataset/MVTec-AD/bottle/train/good/000.png"
    clean = Image.open(path).convert("RGB")
    import numpy as np

    arr = np.asarray(clean, dtype=np.float32) / 255.0
    corrupted_a, mask_a = synthetic_defect(arr, "key")
    corrupted_b, mask_b = synthetic_defect(arr, "key")

    assert mask_a.any()
    assert np.array_equal(mask_a, mask_b)
    assert np.allclose(corrupted_a, corrupted_b)


def test_build_ablation_records_non_diffusion_blocker(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    rows, summary_rows, summary = build_ablation(tmp_path, tmp_path / "out")

    assert len(rows) == 30
    assert len(summary_rows) == 10
    assert summary["images"] == 3
    assert summary["release_gate_passed"] is False
    assert summary["diffusion_executor_ready"] is False
    assert summary["executor_family_coverage"]["partial_conv_proxy_ready"] is True
    assert summary["executor_family_coverage"]["trained_light_ae_ready"] is True
    assert summary["executor_family_coverage"]["trained_light_unet_or_partial_conv_ready"] is True
    assert summary["executor_family_coverage"]["light_unet_proxy_ready"] is True
    assert summary["lpips_metric_protocol"] == "lpips_proxy_l1_only"
    assert summary["perceptual_proxy_metric"] == "lpips_proxy_l1"
    assert "same-protocol diffusion" in summary["required_for_release"][0]


def test_build_ablation_can_record_lpips_metric_with_injected_scorer(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    rows, summary_rows, summary = build_ablation(
        tmp_path,
        tmp_path / "out",
        lpips_scorer=lambda clean, repaired: 0.123,
    )

    assert rows
    assert all(row["lpips_available"] for row in rows)
    assert all(row["lpips"] == 0.123 for row in rows)
    assert all(row["lpips_available"] for row in summary_rows)
    assert summary["lpips_available"] is True
    assert summary["lpips_metric_protocol"] == "external_lpips_scorer"
    assert summary["executor_family_coverage"]["lpips_metric_ready"] is True
    assert "LPIPS metric" not in " ".join(summary["required_for_release"])


def test_build_ablation_can_record_same_protocol_diffusion_executor(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    def perfect_diffusion(image_info, corrupted, mask):
        clean = Image.open(image_info["path"]).convert("RGB").resize((128, 128))
        import numpy as np

        clean_arr = np.asarray(clean, dtype=np.float32) / 255.0
        repaired = corrupted.copy()
        repaired[mask] = clean_arr[mask]
        return repaired

    rows, summary_rows, summary = build_ablation(
        tmp_path,
        tmp_path / "out",
        lpips_scorer=lambda clean, repaired: 0.01,
        diffusion_executor=perfect_diffusion,
        diffusion_coverage_scope="injected_complete",
    )

    assert any(row["executor"] == DIFFUSION_EXECUTOR_NAME for row in rows)
    assert any(row["executor"] == DIFFUSION_EXECUTOR_NAME for row in summary_rows)
    assert summary["diffusion_executor_ready"] is True
    assert summary["executor_family_coverage"]["same_protocol_diffusion_ready"] is True
    assert summary["diffusion_pareto_decision"]["ready"] is True
    assert summary["required_for_release"] == []
    assert summary["release_gate_passed"] is True


def test_diffusion_checkpoint_registry_finds_existing_run_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/feature_mvtec15_bottle_models/diffusion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    registry = DiffusionCheckpointRegistry(checkpoint_root=tmp_path / "runs")

    found = registry.checkpoint_for({"dataset": "mvtec15", "category": "bottle"})

    assert found == checkpoint


def test_write_outputs_creates_tables_and_case_panels(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    out_dir = tmp_path / "tables/repair_executor_ablation"

    summary = write_outputs(tmp_path, out_dir, images_per_category=1)

    assert summary["categories"] == 3
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_repair_executor_images.csv").is_file()
    assert (out_dir / "table_repair_executor_summary.csv").is_file()
    assert (out_dir / "table_executor_family_coverage.csv").is_file()
    assert summary["before_after_visualizations"] >= 1
