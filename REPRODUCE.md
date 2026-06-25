# Reproducing The Claim-Bounded Paper Package

This repository's paper-facing route is feature-first and label-free:
normal images plus deterministic synthetic defects select the pixel policy and
fixed threshold. Real anomaly labels and masks are reserved for final held-out
evaluation, oracle diagnostics, and post-hoc failure analysis.

## 1. Environment

Use one of the pinned entry points:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

or:

```powershell
conda env create -f environment.yml
conda activate lite-seer-ad
```

The default test command must work without manually setting `PYTHONPATH`:

```powershell
pytest -q
```

## 2. Main Artifact Regeneration

Regenerate the strict fixed-threshold paper tables and figures:

```powershell
python tools/export_strict_threshold_paper_artifacts.py
```

Regenerate the seven-baseline comparison:

```powershell
python tools/export_external_baseline_comparison.py
```

Regenerate the image-score aggregation audit:

```powershell
python tools/select_image_score_aggregation.py
```

Regenerate the weak-category failure taxonomy:

```powershell
python tools/export_failure_taxonomy.py
```

When final author, funding, conflict, and availability information is known,
copy `submission_metadata.template.json` to `submission_metadata.json`, replace
all placeholder values, and render the final statement page:

```powershell
python tools/render_submission_statements.py `
  --input submission_metadata.json `
  --out docs/submission_statement_placeholders.md
```

When public release identifiers are known, copy
`release_metadata.template.json` to `release_metadata.json`, replace all
placeholder values, and render the release metadata consumed by the public
release gate:

```powershell
python tools/render_release_metadata.py --input release_metadata.json
```

Build a SHA256 manifest for release files:

```powershell
python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json
```

Export the strict fixed-threshold JSON bundle:

```powershell
python scripts/release/export_fixed_threshold_bundle.py
```

Build the selected prediction-array manifest:

```powershell
python scripts/release/build_prediction_array_manifest.py `
  --out artifacts/predictions_manifest.json
```

Export the public-release and submission-package readiness audits:

```powershell
python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded
python tools/export_second_hardware_run_package.py
python tools/export_release_readiness.py
python tools/export_claim_traceability.py
python tools/export_deployment_readiness.py
python tools/export_submission_package_readiness.py
python tools/export_final_external_handoff.py
python tools/export_final_input_packet.py
python tools/export_submission_finalization_package.py
python tools/export_final_100_closeout_package.py
python tools/validate_release_submission_consistency.py
python tools/validate_final_100_closeout.py
python tools/validate_final_manifest_coverage.py
```

To clear the final cross-hardware deployment gate, run one production-style
probe on a second machine, then collect its hardware profile. The preferred
path is to generate the second-hardware run package and copy
`tables/deployment_production_latency/second_hardware_run_package/` to the
second machine:

```powershell
python tools/export_second_hardware_run_package.py
```

On the second machine, run:

```powershell
.\run_second_hardware_probe.ps1
```

The manual equivalent is:

```powershell
python tools/measure_production_energy.py `
  --output tables/deployment_production_latency/energy_measurements/<second_hardware_probe>.json `
  --label <second_hardware_label> `
  -- python infer.py --config configs/mvtec.yaml --category bottle `
    --checkpoint runs/fulltest_mvtec15_bottle_models/diffusion.pt `
    --sev-checkpoint runs/fulltest_mvtec15_bottle_models/hn_sev.pt `
    --feature-prior-checkpoint runs/fulltest_mvtec15_bottle_models/feature_prior.pt `
    --run-name <second_hardware_probe_run> --ablation utility_lc_rds `
    --latency-budget-ms 10 --image-size 128 --max-samples 8 --seed 7 `
    --device auto --image-score-mode top5 `
    --image-score-source feature_raw_cosine --pixel-heatmap-source feature_raw `
    --reconstruction-steps 5

python tools/collect_hardware_profile.py `
  --output tables/deployment_production_latency/hardware_profiles/<second_hardware>.json `
  --label <second_hardware_label> `
  --validation-run runs/<second_hardware_probe_run>
```

Before accepting the returned files into the release evidence, validate and
stage them on the main workspace. If using the generated package, run
`.\validate_returned_second_hardware_package.ps1`; the manual equivalent is:

```powershell
python tools/validate_second_hardware_package.py `
  --hardware-profile <second_hardware_profile.json> `
  --energy-measurement <second_hardware_energy.json> `
  --stage
```

## 3. Statistics

Use paired statistics for paper claims. Example:

```powershell
python scripts/eval/paired_bca_stats.py `
  --csv tables/external_baseline_comparison/table_lite_vs_external_baselines.csv `
  --where-column method `
  --where-equals padim `
  --method-column lite_aupro `
  --baseline-column external_aupro `
  --metric-name aupro `
  --out-json artifacts/stats/lite_vs_padim_aupro.json
```

The script reports the paired mean delta, BCa confidence interval, wins,
losses, ties, and paired sign-test p value.

## 4. Claim Boundaries

Use only `synthetic_normal_fixed_threshold_v1` fixed Dice/F1/IoU in the main
paper. `oracle_*` metrics are diagnostic upper bounds for the supplement.
CRV remains a repair visualization and post-hoc audit because its pooled
SDR-GT alignment is negative. DiffusionAD full training is optional and must
not be replaced by smoke or partial checkpoints.

## 5. Release Checklist

Before tagging `v1.0-paper`, verify:

- `pytest -q` passes in a clean environment.
- At least one main table is regenerated from public artifacts.
- `artifacts/manifest.json` contains SHA256 hashes for release files.
- `tables/release_readiness/summary.json` reports
  `local_artifact_ready=true`.
- The paper, model card, dataset card, and Zenodo metadata agree on the same
  claim boundary.
- After GitHub Release, Zenodo, and Hugging Face publication, copy
  `release_metadata.template.json` to `release_metadata.json`, replace the
  public identifiers and creator metadata, then run
  `python tools/render_release_metadata.py --input release_metadata.json` and
  `python tools/export_release_readiness.py`.
- After final author, funding, conflict, and cover-letter metadata are known,
  copy `submission_metadata.template.json` to `submission_metadata.json`,
  replace all placeholders, generate
  `tables/submission_package_readiness/final_upload_closeout_package/`, and run
  its `finalize_submission_upload.ps1` script. The first step validates that
  `release_metadata.json` and `submission_metadata.json` use the same public
  identifiers.
- After second-hardware evidence, public release metadata, and submission
  metadata are all available, generate
  `tables/final_external_handoff/final_100_closeout_package/` and run
  `finalize_100_percent.ps1` with the returned second-hardware JSON paths.
  The final script runs `python tools/validate_final_100_closeout.py
  --fail-on-incomplete`; only claim 100% if that validation passes.
- Use `tables/final_external_handoff/final_input_packet/` as the field-level
  checklist for the required `release_metadata.json`, `submission_metadata.json`,
  `second_hardware_profile.json`, and `second_hardware_energy.json` inputs.
