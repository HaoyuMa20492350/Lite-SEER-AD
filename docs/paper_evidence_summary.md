# Lite-SEER-AD v2 Paper Evidence Summary

> Historical diffusion-first snapshot retained for traceability. Do not use
> its metrics as the current paper result; use
> `tables/feature_first_fusion_aggregate_paper_package/`.

## Current State

- MVTec15 plan audit: `complete=true` was previously verified by `tables/plan_audit/plan_audit.json`.
- Baseline coverage: `120/120`, complete=`True`.
- MVTec5 gate: HN-SEV `5/5`, CRV `5/5`, LC-RDS `5/5`.
- MVTec15 gate: HN-SEV `12/15`, CRV `13/15`, LC-RDS `14/15`.

## MVTec15 Reading

| method | image_auroc | pixel_auroc | aupro | pixel_ap | dice |
| --- | --- | --- | --- | --- | --- |
| padim | 0.9095 | 0.9560 | 0.8314 | 0.4299 | 0.4911 |
| patchcore | 0.8854 | 0.9568 | 0.8323 | 0.4193 | 0.4787 |
| rd4ad | 0.6761 | 0.8440 | 0.5194 | 0.2024 | 0.2441 |
| simplenet | 0.7586 | 0.8017 | 0.5690 | 0.1877 | 0.2509 |
| draem | 0.4895 | 0.6638 | 0.3753 | 0.1140 | 0.1636 |
| lite_seer_ad_crv05 | 0.4275 | 0.6212 | 0.3333 | 0.0676 | 0.1135 |
| lite_seer_ad_crv035 | 0.4242 | 0.6225 | 0.3350 | 0.0671 | 0.1133 |
| uniad | 0.5821 | 0.5588 | 0.2450 | 0.0428 | 0.0860 |
| diffusionad | 0.5384 | 0.5417 | 0.1792 | 0.0369 | 0.0708 |
| ddad | 0.5635 | 0.5469 | 0.1785 | 0.0368 | 0.0707 |

- Against the best baseline by category, `lite_seer_ad_crv035` wins `0/15` categories on Pixel AP.
- Mean Pixel AP delta versus the best category baseline is `-0.3827`.
- This does not support a全面 SOTA claim. The paper should emphasize repair-aware verification, module evidence, and low-budget regional diffusion reasoning.

| category | ours_pixel_ap | best_baseline | best_pixel_ap | delta |
| --- | --- | --- | --- | --- |
| bottle | 0.0855 | patchcore | 0.6806 | -0.5952 |
| cable | 0.0377 | padim | 0.4714 | -0.4337 |
| capsule | 0.0176 | patchcore | 0.3417 | -0.3241 |
| carpet | 0.0206 | padim | 0.5326 | -0.5120 |
| grid | 0.0108 | patchcore | 0.1067 | -0.0958 |
| hazelnut | 0.1762 | padim | 0.5736 | -0.3973 |
| leather | 0.1942 | padim | 0.2916 | -0.0974 |
| metal_nut | 0.1540 | patchcore | 0.7110 | -0.5570 |
| pill | 0.0342 | patchcore | 0.5227 | -0.4885 |
| screw | 0.0126 | padim | 0.1525 | -0.1399 |
| tile | 0.0713 | patchcore | 0.4967 | -0.4255 |
| toothbrush | 0.0228 | patchcore | 0.4814 | -0.4586 |
| transistor | 0.0537 | padim | 0.6972 | -0.6436 |
| wood | 0.0718 | patchcore | 0.3793 | -0.3075 |
| zipper | 0.0438 | padim | 0.3077 | -0.2639 |

## Efficiency Reading

| method | latency_ms | fps | nfe |
| --- | --- | --- | --- |
| uniad | 0.69 | 1452.89 | 0.00 |
| padim | 1.51 | 890.44 | 0.00 |
| patchcore | 2.61 | 537.72 | 0.00 |
| diffusionad | 18.95 | 52.79 | 0.00 |
| rd4ad | 19.48 | 51.37 | 0.00 |
| draem | 23.66 | 42.27 | 0.00 |
| ddad | 38.62 | 25.90 | 0.00 |
| simplenet | 53.02 | 18.87 | 0.00 |
| lite_seer_ad_crv035 | 297.29 | 3.37 | 16.14 |
| lite_seer_ad_crv05 | 300.00 | 3.34 | 16.14 |

- Current Lite-SEER-AD runs are slower than memory/statistical baselines and many local baselines on MVTec15.
- LC-RDS should therefore be presented as a budget-allocation contribution inside the repair-aware diffusion pipeline, not as an overall fastest-method claim.

## Protocol Decisions

- Default CRV weight for paper-scale MVTec15 comparison: `0.35`.
- MVTec5 CRV search recommended `0.5`; keep it as weight-sensitivity evidence, not the default for the full paper protocol.
- Default narrative: repair-aware anomaly verification + low-budget diffusion scheduling, not full-metric SOTA.
- Default next dataset order: VisA first, MPDD second.

## Dataset Smoke Status

- VisA smoke: complete=`True`, runs=`4`, failures=`0`.
- MPDD smoke: complete=`True`, runs=`4`, failures=`0`.
- These smoke runs validate data loading, masks, standard outputs, qualitative artifacts, and summary export only. They are not paper-strength contribution evidence.

## VisA Mini Status

- VisA mini core pass: complete=`True`, schema_runs=`60`, failures=`0`.
- Covered categories: `12`/`12` detected VisA categories: `candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`.
- Full-model mean metrics: image AUROC `0.4504`, pixel AUROC `0.5991`, AUPRO `0.2640`, Pixel AP `0.0033`, Dice `0.0107`.
- Mini module gate: HN-SEV `12/12`, CRV `6/12`, prototype `0/0`, LC-RDS `0/0`.
- This mini pass used the core ablation set: `full,residual_only,no_sev,no_crv,rule_brds`. Prototype and full LC-RDS gates are therefore intentionally incomplete.

## VisA Gate Status

- VisA gate pass: complete=`True`, schema_runs=`72`, failures=`0`.
- Covered categories: `12`/`12` detected VisA categories: `candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`.
- Full-model mean metrics from this pass: image AUROC `0.4504`, pixel AUROC `0.5991`, AUPRO `0.2640`, Pixel AP `0.0033`, Dice `0.0107`.
- Gate-specific module evidence: prototype `6/12`, LC-RDS `5/12`.
- This pass complements the VisA mini core pass. It intentionally does not retest HN-SEV/CRV comparisons, so the gate summary should be read by module family rather than as a single all-module ready flag.

## MPDD Mini Status

- MPDD mini core pass: complete=`True`, schema_runs=`30`, failures=`0`.
- Covered categories: `6`/`6` detected MPDD categories: `bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes`.
- Full-model mean metrics: image AUROC `0.4437`, pixel AUROC `0.5684`, AUPRO `0.2196`, Pixel AP `0.0172`, Dice `0.0338`.
- Mini module gate: HN-SEV `5/6`, CRV `4/6`, prototype `0/0`, LC-RDS `0/0`.
- MPDD now has all-category core evidence. Prototype and full LC-RDS still need a gate-specific pass if MPDD module claims are kept symmetrical with VisA.

## MPDD Gate Status

- MPDD gate pass: complete=`True`, schema_runs=`36`, failures=`0`.
- Covered categories: `6`/`6` detected MPDD categories: `bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes`.
- Full-model mean metrics from this pass: image AUROC `0.4437`, pixel AUROC `0.5684`, AUPRO `0.2196`, Pixel AP `0.0172`, Dice `0.0338`.
- Gate-specific module evidence: prototype `3/6`, LC-RDS `5/6`.
- This completes the same mini/core plus gate-specific evidence pattern now used for VisA.

## Baseline Expansion Status

- The MVTec15 baseline runner now supports non-MVTec config/category discovery and writes dataset-specific table aliases such as `table_main_visa.csv` and `table_main_mpdd.csv`.
- VisA baseline smoke: complete=`True`, methods=`6/6`, failures=`0`.
- MPDD baseline smoke: complete=`True`, methods=`6/6`, failures=`0`.
- VisA full baseline table: complete=`True`, coverage=`72/72`, failures=`0`.
- MPDD full baseline table: complete=`True`, coverage=`36/36`, failures=`0`.
- Smoke baselines used tiny samples and `--allow-random-weights`; full baseline tables were rerun across all categories without random weights.

## Paper Package Status

- Cross-dataset package exported to `tables/paper_package`: run_status_ok=`True`, baseline_coverage_ok=`True`.
- Package artifacts: `table_main_cross_dataset.csv`, `table_efficiency_cross_dataset.csv`, `table_mean_by_dataset_method.csv`, `table_module_gates.csv`, `table_run_status.csv`, `table_baseline_coverage.csv`, and `failure_analysis_notes.md`.
- Results/limitations draft: `docs/results_limitations_draft.md`.

## Next Experiments

- VisA root available: `SEER-AD-dataset/VisA` with `12` detected categories: `candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`.
- MPDD root available: `SEER-AD-dataset/MPDD/official/MPDD/MPDD` with `6` detected categories: `bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes`.
- Next analysis step: inspect `tables/paper_package/table_mean_by_dataset_method.csv` and category-level deltas to write the actual Results/Limitations narrative.
- Next figure step: build Pareto and qualitative failure panels from current MVTec15, VisA, and MPDD runs.

## Acceptance Gates

- VisA: full model all 12 categories, at least 6 baselines, main/efficiency/HN-SEV/CRV/LC-RDS tables, and qualitative cases.
- MPDD: full model all 6 categories, baseline coverage aligned with VisA, reflective/metal false-positive analysis, SDR evidence, and failure cases.
- Paper package: merged MVTec/VisA/MPDD tables, Pareto plots, module ablations, and honest failure analysis.
