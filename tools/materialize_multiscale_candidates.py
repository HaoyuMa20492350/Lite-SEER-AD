from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize fixed multiscale feature candidates without anomaly-label tuning.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", required=True)
    p.add_argument(
        "--primary-checkpoint-template",
        default="runs/feature_highres_prior_mvtec15_{category}_models/feature_prior.pt",
    )
    p.add_argument(
        "--secondary-checkpoint-template",
        default="runs/feature_padim128_resnet18_l123_full_prior_mvtec15_{category}_models/feature_prior.pt",
    )
    p.add_argument("--run-prefix", default="feature_multiscale_mvtec15")
    p.add_argument("--variants", default="raw35,raw55,highpass35,highpass55")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def variant_spec(name: str) -> tuple[float, str]:
    specs = {
        "raw35": (0.35, "raw"),
        "raw55": (0.55, "raw"),
        "raw75": (0.75, "raw"),
        "highpass35": (0.35, "highpass:3"),
        "highpass55": (0.55, "highpass:3"),
        "highpass75": (0.75, "highpass:3"),
    }
    if name not in specs:
        raise ValueError(f"Unknown multiscale variant: {name}. Available: {sorted(specs)}")
    return specs[name]


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "selection_protocol": "predeclared_candidates_for_synthetic_normal_utility",
        "uses_real_anomaly_labels_for_candidate_configuration": False,
        "categories": split_csv(args.categories),
        "variants": split_csv(args.variants),
        "runs": [],
    }
    for category in split_csv(args.categories):
        primary = Path(args.primary_checkpoint_template.format(category=category))
        secondary = Path(args.secondary_checkpoint_template.format(category=category))
        if not primary.exists() or not secondary.exists():
            raise FileNotFoundError(f"Missing multiscale checkpoints for {category}: {primary}, {secondary}")
        for variant in split_csv(args.variants):
            primary_weight, secondary_postprocess = variant_spec(variant)
            run_name = f"{args.run_prefix}_{category}_{variant}"
            output = REPO_ROOT / "runs" / run_name / "predictions.npz"
            if args.resume and output.exists():
                report["runs"].append({"category": category, "variant": variant, "run": str(output.parent), "status": "existing"})
                continue
            command = [
                sys.executable,
                "tools/materialize_feature_prior_candidate.py",
                "--config",
                args.config,
                "--category",
                category,
                "--feature-prior-checkpoint",
                str(primary),
                "--secondary-feature-prior-checkpoint",
                str(secondary),
                "--image-size",
                str(args.image_size),
                "--batch-size",
                str(args.batch_size),
                "--run-name",
                run_name,
                "--multiscale-primary-weight",
                str(primary_weight),
                "--multiscale-secondary-postprocess",
                secondary_postprocess,
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
            if args.max_samples is not None:
                command.extend(["--max-samples", str(args.max_samples)])
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            report["runs"].append({"category": category, "variant": variant, "run": str(output.parent), "status": "created"})
    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
