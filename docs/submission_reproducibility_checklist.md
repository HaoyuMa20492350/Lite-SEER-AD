# Lite-SEER-AD Submission and Reproducibility Checklist

Status date: 2026-06-14.

## Frozen Scientific Scope

- [x] Main contribution is label-free pixel-policy and threshold selection.
- [x] Main detector is separated from HN-SEV, LC-RDS, and CRV module claims.
- [x] CRV is restricted to visualization and post-hoc inspection.
- [x] Test-GT thresholds are reported only as `oracle_*` diagnostics.
- [x] Universal SOTA and MVTec AD 2 claims are excluded from the main paper.
- [x] Negative retrieval, multiscale, and image-score aggregation results are
  retained rather than hidden.

## Main Evidence

- [x] MVTec AD, VisA, and MPDD: 33 categories, seeds `7/13/23`, 99 held-out
  runs.
- [x] Candidate agreement is `100%` across held-out seeds.
- [x] All main runs use `synthetic_normal_fixed_threshold_v1`.
- [x] Maximum observed normal-pixel FPR is below the frozen `0.5%` cap.
- [x] Strict tables are in `tables/strict_fixed_threshold/`.
- [x] Paper tables and threshold figures are in
  `tables/strict_fixed_threshold_paper/`.
- [x] The authoritative main package is
  `tables/feature_first_fusion_aggregate_paper_package/`.

## External Comparisons

- [x] PatchCore, PaDiM, UniAD, DRAEM, DDAD, RD4AD, and SimpleNet cover all 15
  MVTec AD categories under one metric and threshold implementation.
- [x] Source commits and provenance are frozen in
  `baselines/official_sources.json`.
- [x] Category-paired bootstrap confidence intervals, exact sign tests, Holm
  correction, and worst-category attribution are exported to
  `tables/external_baseline_comparison/`.
- [x] The manuscript limits robust superiority to the supported PaDiM Pixel
  AUROC result.
- [ ] DiffusionAD full 15-category author-configuration training is pending.
  Its smoke run is not paper eligible, and partial-training checkpoints and
  predictions are not retained.
- [x] The measured DiffusionAD compute plan is in
  `tables/diffusionad_compute_plan/`: 663,000 optimizer steps and about
  3,801 aggregate GPU hours on the available 16-GB execution path.

## Negative and Failure Evidence

- [x] The 11-mode label-free image-score aggregation audit is complete.
- [x] The frozen decision is `retain_current_top5`; no post-hoc switching rule
  is fitted from held-out labels.
- [x] Aggregation evidence is in
  `tables/image_score_aggregation_mvtec15/`.
- [x] The current frozen failure panel is
  `paper/figures/fig_frozen_failure_cases.png`.
- [x] Its GT-aware diagnostic case index and supervision boundary are in
  `tables/failure_case_panel_mvtec15/`.
- [x] The broader weak-category taxonomy is in `tables/failure_taxonomy/`;
  it is post-hoc diagnostic evidence and is not used for method selection.

## Optional MVTec AD 2 Archive

- [x] Dataset layout passes local validation.
- [x] Public inference is complete for 8 categories and 3 seeds.
- [x] Private and private-mixed export covers 4090 images.
- [x] The official checker passes.
- [x] Upload archive:
  `submissions/mvtec_ad2_seed7_model256.tar.gz`.
- [x] Archive SHA256:
  `9ae82e41018ad17a59c5f9ae176f4a12355cef9f7bc9bc143e342145deff266b`.
- [x] AD2 is excluded from the main paper and is not a release gate.
- [ ] Optional future action: upload when authenticated server access is
  available.
- [ ] Optional future action: import the server result and audit the same-date
  leaderboard.

## Paper Package

- [x] Manuscript draft: `paper/manuscript.md`.
- [x] Supplement draft: `paper/supplement.md`.
- [x] References: `paper/references.bib`.
- [x] Main result, fixed-vs-oracle, threshold audit, module, repair, and failure
  figures are available.
- [x] Claim boundaries and limitations are synchronized across
  `docs/paper_protocol.md`, `docs/results_limitations_draft.md`, and the
  manuscript.
- [ ] Convert to the target journal's LaTeX or Word template after the journal
  is selected.
- [ ] Apply the target journal's word, figure, reference-style, and
  supplementary-file limits.
- [ ] Add author identities, affiliations, acknowledgements, funding,
  conflicts, and data/code availability statements.
- [ ] When those final values are known, copy
  `submission_metadata.template.json` to `submission_metadata.json`, replace
  all placeholder values, then render
  `docs/submission_statement_placeholders.md` with
  `python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md`.
- [x] Submission-package readiness audit:
  `tables/submission_package_readiness/`.
- [x] Final external-action handoff:
  `tables/final_external_handoff/`.
- [x] Field-level final input packet:
  `tables/final_external_handoff/final_input_packet/`.

## Reproduction Commands

Run the local verification suite:

```powershell
pytest -q
```

Regenerate strict paper tables and figures:

```powershell
python tools/export_strict_threshold_paper_artifacts.py
```

Regenerate the seven-baseline comparison and paired inference:

```powershell
python tools/export_external_baseline_comparison.py
```

Regenerate the label-free image-score audit without recomputing model
evidence:

```powershell
python tools/select_image_score_aggregation.py
```

Regenerate the frozen failure-case panel:

```powershell
python tools/export_failure_case_panel.py
```

Regenerate the weak-category failure taxonomy:

```powershell
python tools/export_failure_taxonomy.py
```

Render final submission statements after author and release metadata are
available:

```powershell
python tools/render_submission_statements.py `
  --input submission_metadata.json `
  --out docs/submission_statement_placeholders.md
```

Render final public-release metadata after GitHub, Zenodo, and Hugging Face
identifiers are available:

```powershell
python tools/render_release_metadata.py --input release_metadata.json
```

Generate the final-upload closeout package after `release_metadata.json` and
`submission_metadata.json` are ready:

```powershell
python tools/export_submission_finalization_package.py
python tools/validate_release_submission_consistency.py
```

Generate the final 100% closeout package after second-hardware evidence,
release metadata, and submission metadata are available:

```powershell
python tools/export_final_100_closeout_package.py
python tools/export_final_input_packet.py
python tools/validate_final_100_closeout.py
python tools/validate_final_manifest_coverage.py
```

Export release metadata and file hashes:

```powershell
python scripts/release/export_fixed_threshold_bundle.py
python scripts/release/build_prediction_array_manifest.py `
  --out artifacts/predictions_manifest.json
python scripts/release/build_artifact_manifest.py `
  --out artifacts/manifest.json
python tools/export_claim_traceability.py
python tools/export_final_external_handoff.py
python tools/export_final_input_packet.py
```

Validate returned second-hardware deployment evidence before staging it:

```powershell
python tools/export_second_hardware_run_package.py

python tools/validate_second_hardware_package.py `
  --hardware-profile <second_hardware_profile.json> `
  --energy-measurement <second_hardware_energy.json> `
  --stage
```

Run paired paper statistics from a long-form baseline table:

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

Refresh readiness summaries:

```powershell
python tools/audit_official_baseline_readiness.py --datasets mvtec15
python tools/export_plan_acceptance.py
python tools/export_lc_rds_budget_sweep.py
python tools/export_repair_executor_ablation.py --enable-lpips
python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded
python tools/export_deployment_readiness.py
python tools/export_submission_package_readiness.py
```

To clear the remaining cross-hardware deployment gate, repeat a production
probe on a second hardware profile, then collect its profile sidecar:

```powershell
python tools/measure_production_energy.py `
  --output tables/deployment_production_latency/energy_measurements/<second_hardware_probe>.json `
  --label <second_hardware_label> `
  -- python infer.py `
    --config configs/mvtec.yaml `
    --category bottle `
    --checkpoint runs/fulltest_mvtec15_bottle_models/diffusion.pt `
    --sev-checkpoint runs/fulltest_mvtec15_bottle_models/hn_sev.pt `
    --feature-prior-checkpoint runs/fulltest_mvtec15_bottle_models/feature_prior.pt `
    --run-name <second_hardware_probe_run> `
    --ablation utility_lc_rds `
    --latency-budget-ms 10 `
    --image-size 128 `
    --max-samples 8 `
    --seed 7 `
    --device auto `
    --image-score-mode top5 `
    --image-score-source feature_raw_cosine `
    --pixel-heatmap-source feature_raw `
    --reconstruction-steps 5

python tools/collect_hardware_profile.py `
  --output tables/deployment_production_latency/hardware_profiles/<second_hardware>.json `
  --label <second_hardware_label> `
  --validation-run runs/<second_hardware_probe_run>

python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded
python tools/export_deployment_readiness.py
python tools/export_completion_gap_matrix.py
```

Run the remaining full DiffusionAD reproduction only after remote parallel
capacity is configured:

```powershell
python tools/run_official_diffusionad.py `
  --categories all `
  --epochs 3000 `
  --batch-size 16 `
  --micro-batch-size 4 `
  --amp `
  --resume
```

Optional, after receiving a future MVTec AD 2 server export:

```powershell
python tools/import_mvtec_ad2_official_results.py `
  --input "<SERVER_RESULT.json>" `
  --source-url "https://benchmark.mvtec.com/" `
  --submission-id "<SUBMISSION_ID>" `
  --evaluated-at "<ISO_DATETIME>" `
  --update-draft
```

## Release Gate

The paper package is locally reproducible and claim-bounded. Final submission
remains gated by:

1. target-journal selection and template conversion;
2. author, affiliation, funding, conflict, and data/code statements;
3. DiffusionAD full training only if the selected journal or review strategy
   requires the eighth external baseline.
4. completion or explicit limitation text for HN-SEV TP retention/calibration,
   LC-RDS budget violation, and diffusion necessity experiments.
