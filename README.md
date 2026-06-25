# Lite-SEER-AD

Label-free feature-first industrial anomaly localization with selective
regional verification.

The current paper-facing route is:

- Frozen feature-first detection for the main anomaly map.
- Label-free pixel-policy and fixed-threshold selection from normal images and
  deterministic synthetic defects.
- HN-SEV as false-positive suppression evidence.
- LC-RDS as budgeted local repair scheduling evidence.
- CRV only as repair visualization and post-hoc audit, not as detector
  evidence.
- Standardized artifacts for strict metrics, baselines, statistics, and
  reproducibility.

## Paper Package Verification

```powershell
pytest -q
python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json
```

See `REPRODUCE.md`, `MODEL_CARD.md`, `DATASETS.md`, and
`docs/claim_freeze_2026_06_21.md` before preparing a release. The main paper
must use `synthetic_normal_fixed_threshold_v1`; `oracle_*` metrics are
diagnostic upper bounds only.

## Legacy Diffusion Smoke Run

```powershell
python train_diffusion.py --config configs/mvtec.yaml --category bottle --epochs 1 --max-samples 8 --batch-size 2 --image-size 128 --run-name smoke
python mine_hard_negatives.py --config configs/mvtec.yaml --category bottle --checkpoint runs/smoke/diffusion.pt --max-samples 8 --image-size 128 --run-name smoke
python train_hn_sev.py --config configs/mvtec.yaml --category bottle --checkpoint runs/smoke/diffusion.pt --hard-negative-dir runs/smoke/hard_negatives --epochs 1 --max-samples 8 --batch-size 4 --image-size 128 --run-name smoke
python train_lc_rds.py --config configs/mvtec.yaml --run-name smoke
python infer.py --config configs/mvtec.yaml --category bottle --checkpoint runs/smoke/diffusion.pt --sev-checkpoint runs/smoke/hn_sev.pt --scheduler-checkpoint runs/smoke/lc_rds.pt --max-samples 8 --image-size 128 --run-name smoke
python evaluate.py --predictions runs/smoke/predictions.npz --out runs/smoke/eval_metrics.json
```

The smoke run is intentionally tiny. It verifies that the legacy repair
pipeline works end to end; it is not paper evidence and must not be reported as
DiffusionAD or as the main detector.

## Historical MVTec5 Evidence Loop

This older phase is retained for traceability on five MVTec classes:
`bottle,cable,capsule,metal_nut,zipper`.

```powershell
# Inspect the generated command plan without running jobs.
python tools/run_next_phase_mvtec5.py --smoke --dry-run

# Tiny executable loop for plumbing checks.
python tools/run_next_phase_mvtec5.py --smoke --resume

# Full 5-class evidence loop at 128px/64 samples by default.
python tools/run_next_phase_mvtec5.py --resume
```

This runner keeps new results under the `next_mvtec5_*` prefix, runs a
`reconstruction_steps` sweep, trains HN-SEV variants, searches CRV fusion
weights, exports qualitative cases, and writes:

- `tables/next_mvtec5/table_reconstruction_sweep.csv`
- `tables/next_mvtec5/table_ablation_hn_sev.csv`
- `tables/next_mvtec5/table_ablation_crv.csv`
- `tables/next_mvtec5/table_ablation_lc_rds.csv`
- `tables/next_mvtec5/table_crv_weight_search.csv`
- `tables/next_mvtec5/gate_summary.json`

Only proceed to MVTec15 when `gate_summary.json` reports
`ready_for_mvtec15: true`. The gate requires category-level evidence for
HN-SEV, the prototype sub-branch, and LC-RDS. CRV deltas are now interpreted
only as post-hoc repair diagnostics.

## Feature-First Architecture Flip

The current main metric path is feature-first detection with regional modules
audited separately. Train a frozen-feature normality prior, then run the
feature-first MVTec5 flip experiment:

```powershell
# Plumbing check only; random feature weights are not paper evidence.
python tools/run_feature_first_mvtec5.py --smoke --resume --allow-random-feature-weights

# Paper-facing MVTec5 flip; requires cached/network ImageNet weights.
python tools/run_feature_first_mvtec5.py --resume

# Full MVTec15 expansion.
python tools/run_feature_first_mvtec5.py --categories bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper --run-prefix feature_mvtec15 --tables-out tables/feature_mvtec15 --resume

# Cross-dataset smoke checks; random feature weights are plumbing only.
python tools/run_feature_first_mvtec5.py --config configs/visa.yaml --categories auto --run-prefix feature_visa --tables-out tables/feature_visa --smoke --resume --allow-random-feature-weights
python tools/run_feature_first_mvtec5.py --config configs/mpdd.yaml --categories auto --run-prefix feature_mpdd --tables-out tables/feature_mpdd --smoke --resume --allow-random-feature-weights

# Cross-dataset paper-facing runs; requires ImageNet pretrained features.
python tools/run_feature_first_mvtec5.py --config configs/visa.yaml --categories auto --run-prefix feature_visa --tables-out tables/feature_visa --resume
python tools/run_feature_first_mvtec5.py --config configs/mpdd.yaml --categories auto --run-prefix feature_mpdd --tables-out tables/feature_mpdd --resume

# Export the latest feature-first cross-dataset comparison package.
python tools/export_feature_first_package.py --out tables/feature_paper_package

# Diagnose Pixel AP/Dice weak classes without retraining the full pipeline.
python tools/materialize_train_normal_pixel_calibration.py --source-run-dir runs/feature_pixelraw_mvtec15_grid_feature_tuned_crv --out-root tables/pixel_calibration/mvtec15_grid --source-map heatmaps --normal-source-map feature_raw
python tools/search_pixel_postprocess.py --run-dir tables/pixel_calibration/mvtec15_grid/feature_pixelcal_heatmaps_relu_robust_zscore --out tables/pixel_postprocess/mvtec15_grid_cal_relu_robust --materialize

# Evaluate a fixed train-normal pixel policy without test-label mode selection.
python tools/materialize_fixed_pixel_policy.py --source-run-dir runs/feature_pixelraw_mvtec15_grid_feature_tuned_crv --out-run-dir runs/feature_fixedpixel_mvtec15_grid_feature_pixel_policy --calibration-mode relu_robust_zscore --postprocess-mode highpass:9
```

The runner trains `feature_prior.pt`, feature-aware HN-SEV, keeps legacy CRV
fusion diagnostics for audit, materializes `feature_tuned_crv`, and runs:

- `residual_only`
- `feature_only`
- `feature_hn_sev`
- `feature_hn_sev_crv`
- `feature_tuned_crv`
- `feature_fixed10`
- `feature_fixed25`
- `feature_rule_brds`
- `utility_lc_rds`

Outputs are summarized under `tables/feature_mvtec5/` and
`tables/feature_mvtec15/`. The feature-first runner now uses raw cosine-gap
feature maps for image-level scores (`--image-score-source
feature_raw_cosine`) with `top5` pooling, while pixel metrics use raw feature
heatmaps (`--pixel-heatmap-source feature_raw`) to avoid CRV verifier maps
diluting localization. The current gate reading, raw-score results, pixel
heatmap source diagnostics, and fusion diagnostics are summarized in
`docs/feature_first_evidence_summary.md`.
Smoke runs automatically write to `*_smoke` table directories to keep their CRV
search artifacts separate from full paper-facing runs.

## Label-Free Paper Policy Selection

The paper-facing selector is `synthetic_normal_utility`. It evaluates each
candidate on held-out normal images, deterministic synthetic defect masks,
normal false-positive rate, augmentation stability, and measured latency. It
does not use real anomaly labels or masks to choose a policy.

```powershell
# Materialize one candidate's label-free selection evidence.
python tools/materialize_synthetic_normal_validation.py `
  --candidate-run-dir runs/feature_highres_prior_mvtec15_transistor_feature_tuned_crv

# Run the unified selector after every candidate has synthetic evidence.
python tools/select_pixel_policy_with_normal_gate.py `
  --dataset mvtec15 `
  --categories transistor `
  --candidate 'pixelraw=runs/feature_pixelraw_mvtec15_{category}_feature_tuned_crv' `
  --candidate 'highres256=runs/feature_highres_prior_mvtec15_{category}_feature_tuned_crv' `
  --candidate 'post_highres_gaussian3=runs/feature_post_highres_mvtec15_{category}/feature_pixelpost_gaussian3' `
  --candidate 'fixed=runs/feature_fixedpixel_mvtec15_{category}_feature_pixel_policy' `
  --candidate 'padim128r18l123=runs/feature_padim128_resnet18_l123_full_prior_mvtec15_{category}_feature_tuned_crv' `
  --out tables/synthetic_gate_mvtec15_transistor `
  --gate-metric synthetic_normal_utility `
  --materialize
```

Use `tools/run_synthetic_normal_gate.py` to materialize and select across seeds
`7,13,23`. The older `select_pixel_policy_with_val_split.py` selector reads
real anomaly masks and is explicitly an oracle upper bound; its outputs must
not be reported as the main method.

### Retrieval repair and weak-category candidates

The feature prior can carry a portable normal-patch retrieval bank. Use
`tools/augment_feature_prior_with_retrieval.py` to add one to an existing
checkpoint, then pass `--enable-retrieval-repair` to `infer.py`. Retrieval is
opt-in because the first controlled `transistor` ablation is negative; see
`tables/retrieval_ablation_transistor/`.

Predeclared multiscale candidates can be generated with
`tools/materialize_multiscale_candidates.py`. They are compatible with
`synthetic_normal_utility`, but the first five weak-category gate selected none
of them over the existing highres/PaDiM pool. They remain negative ablations.

`ExpectedUtilityScheduler` now accumulates measured ROI latency, updates
conservative action-latency estimates online, and logs expected gain, expected
latency, realized latency, cumulative spend, and gain per millisecond.

## Main Artifacts

Each run writes standardized outputs under `runs/<run-name>/`:

- `diffusion.pt`, `hn_sev.pt`, `lc_rds.pt`
- `metrics.json`, `eval_metrics.json`, `scores.csv`
- `predictions.npz`
- `heatmaps/`, `masks/`, `repairs/`
- `roi_budget.jsonl`, `pareto.csv`
- `images/<case>/input.png`, `reconstruction.png`, `residual.png`,
  `candidate_roi.png`, `verified_roi.png`, `final_heatmap.png`,
  `pixel_eval_heatmap.png`, `final_mask.png`, `repair.png`,
  `ground_truth.png`

## Notes

MVTec AD 2 is retained as an optional supplementary asset. Its authenticated
server upload and private result are not part of the main MVTec AD, VisA, and
MPDD paper protocol and do not gate submission.

The code is structured for research iteration rather than maximum performance. Start with the smoke run, then scale category count, image size, epochs, and baselines once the MVP behavior is validated.

The bundled baseline runners are explicitly local controls:
`PatchCore-Local`, `PaDiM-Local`, `SimpleNet-Lite`, `DRAEM-Lite`,
`RD4AD-Lite`, `UniAD-Lite`, `DiffusionAD-Lite`, and `DDAD-Lite`. They are not
official reproductions and must not be presented as external SOTA evidence.
Third-party prediction artifacts can still be imported through
`tools/import_external_baseline.py` with exact provenance; use
`tools/audit_mvtec15_baseline_coverage.py` to verify the MVTec15 baseline table
is complete before treating it as publishable.

The strict external MVTec15 matrix currently has `105/120` ready entries:
PatchCore, PaDiM, UniAD, DRAEM, DDAD, RD4AD, and SimpleNet are complete for
all 15 categories. DiffusionAD source, assets, and a real one-batch smoke test
are validated, and its measured compute plan is complete. Partial-training
checkpoints and predictions are not retained. The full
3000-epoch-per-category reproduction requires about 3,801 aggregate GPU hours
on the available 16-GB path. See `docs/current_stage_and_next_plan_zh.md`.
