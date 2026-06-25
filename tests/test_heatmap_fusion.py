import numpy as np

from seer_ad_v2.evaluation.heatmap_fusion import (
    apply_normal_scale,
    fuse_heatmaps,
    normal_scale,
    resize_heatmaps,
)


def test_normal_scale_uses_fixed_normal_statistics() -> None:
    clean = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    center, scale = normal_scale(
        clean, center_quantile=0.5, upper_quantile=0.9
    )
    transformed = apply_normal_scale(clean, center, scale)
    assert np.isclose(np.quantile(transformed, 0.5), 0.0, atol=1e-6)
    assert np.isclose(np.quantile(transformed, 0.9), 1.0, atol=1e-6)


def test_fusion_resizes_to_target_and_respects_weights() -> None:
    source_a = np.full((2, 4, 4), 4.0, dtype=np.float32)
    source_b = np.full((2, 2, 2), 2.0, dtype=np.float32)
    fused = fuse_heatmaps(
        source_a,
        source_b,
        weight_a=0.75,
        scale_a=(0.0, 4.0),
        scale_b=(0.0, 2.0),
        target_shape=(2, 2),
    )
    assert fused.shape == (2, 2, 2)
    assert np.allclose(fused, 1.0)
    assert resize_heatmaps(source_b, (2, 2)).dtype == np.float32
