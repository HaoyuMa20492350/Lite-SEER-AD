import csv

from scripts.release.export_fixed_threshold_bundle import build_threshold_bundle


def test_threshold_bundle_records_label_free_flags(tmp_path):
    csv_path = tmp_path / "strict.csv"
    rows = [
        {
            "dataset": "mvtec15",
            "category": "bottle",
            "split_seed": "seed7",
            "threshold": "0.5",
            "threshold_protocol": "synthetic_normal_fixed_threshold_v1",
            "selected_candidate": "highres256",
            "normal_pixel_fpr": "0.001",
            "heldout_run": "tables/run",
            "uses_real_anomaly_labels_for_threshold": "False",
            "uses_real_anomaly_masks_for_threshold": "False",
        }
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bundle = build_threshold_bundle(csv_path)

    assert bundle["policy_count"] == 1
    assert bundle["uses_real_anomaly_labels"] is False
    assert bundle["uses_real_anomaly_masks"] is False
    assert bundle["policies"][0]["threshold"] == 0.5
