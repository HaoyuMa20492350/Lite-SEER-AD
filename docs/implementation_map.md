# Lite-SEER-AD v2 Implementation Map

## Research Claims to Code Paths

| Claim | Code path | Primary evidence |
|---|---|---|
| Lightweight normal diffusion reconstruction | `train_diffusion.py`, `seer_ad_v2/models/diffusion/` | residual heatmaps, image/pixel metrics |
| HN-SEV suppresses normal high-residual false positives | `mine_hard_negatives.py`, `train_hn_sev.py` | FPRR-style normal ROI reduction, ablation vs synthetic-only |
| CRV makes repair a verification signal | `infer.py`, `repair_verification.py` | score drop, Pixel AP/AUPRO/Dice gain |
| LC-RDS improves latency/NFE tradeoff | `train_lc_rds.py`, `lc_rds.py`, `pareto.csv` | Pareto curve vs fixed/rule budgets |
| Resolution decoupling | `local_refiner.py` | native/local ROI repair and boundary metrics |
| Feature-first detection prior | `train_feature_prior.py`, `seer_ad_v2/models/feature_prior.py` | feature heatmaps, feature ROI proposals, feature-vs-residual flip |
| Multi-space CRV | `infer.py`, `repair_verification.py` | pixel/feature/prototype score drops and `repair_gain` |
| Retrieval-conditioned repair | `feature_prior.py`, `local_refiner.py`, `augment_feature_prior_with_retrieval.py` | retrieval index/similarity/weight logs and fixed-action ablation |
| Label-free candidate selection | `synthetic_validation.py`, `run_synthetic_normal_gate.py` | synthetic utility, normal FPR, stability, latency |
| Measured-latency utility scheduling | `lc_rds.py`, `infer.py` | expected/realized latency and cumulative ROI budget |

## Minimum Publishable Experiment Ladder

1. Smoke: MVTec `bottle`, tiny image size, one epoch.
2. Core evidence: gated MVTec 5 classes with HN-SEV, CRV, and LC-RDS ablations.
3. Main: MVTec AD 15 classes after the MVTec5 gate passes.
4. Strengthening: MVTec AD 2, 3 seeds, few-shot 8/16/32, failure cases.

## Feature-First Strengthening Path

Current results show the diffusion-first detector is not competitive with
PatchCore/PaDiM/SimpleNet on main detection metrics. The active implementation
path is therefore:

```text
frozen feature prior -> feature ROI proposals -> feature-aware HN-SEV
-> local diffusion repair -> multi-space CRV repair_gain
-> tuned CRV verification
-> raw feature pixel heatmap + raw cosine-gap top5 image-score pooling
-> utility LC-RDS
```

Use `tools/run_feature_first_mvtec5.py` for this gate and expansion. The smoke
form is:

```powershell
python tools/run_feature_first_mvtec5.py --smoke --resume --allow-random-feature-weights
```

The paper-facing form omits `--allow-random-feature-weights` so that ImageNet
pretrained features are used. Full MVTec15 evidence is stored under
`tables/feature_mvtec15/`; `feature_tuned_crv` is the current main protocol.
Pixel metrics are now explicitly decoupled from verifier fusion through
`pixel_heatmap_source`; the current fixed global protocol uses `feature_raw`
for localization and `feature_raw_cosine + top5` for image-level scoring.
Raw distance/cosine maps and train-normal fused scoring tools remain available
for ablation and follow-up strengthening.

For Pixel AP/Dice strengthening, use:

- `tools/materialize_fixed_pixel_policy.py` to evaluate one fixed
  train-normal calibration/post-processing policy across categories without
  test-set mode selection.
- `tools/materialize_train_normal_pixel_calibration.py` for train-normal
  spatial z/ratio calibration of saved pixel maps.
- `tools/search_pixel_postprocess.py` for Gaussian, local high-pass, top-hat,
  and closing diagnostics.
- `tools/select_pixel_policy_with_val_split.py` to select among saved pixel
  policy candidates on a deterministic validation split and report held-out
  metrics without test-label policy selection.
- `tools/select_pixel_policy_with_normal_gate.py` to evaluate policy gates that
  use only normal validation images or candidate priors, while keeping the same
  held-out reporting schema.
- `tools/materialize_synthetic_normal_validation.py` to create deterministic
  normal-plus-synthetic evidence for each candidate without real anomaly masks.
- `tools/run_synthetic_normal_gate.py` to run the paper-facing selector for
  seeds `7,13,23`.
- `tools/export_heldout_pixel_policy_package.py` to combine the held-out
  selection reports across datasets.
- `tools/export_heldout_sota_comparison.py` to align selected held-out samples
  with local baseline `predictions.npz` files by path and recompute same-split
  SOTA comparison metrics.
- `tools/run_feature_first_mvtec5.py --selector-only --skip-figures` for
  faster highres candidate generation when only the held-out selector needs
  `feature_hn_sev_crv` and `feature_tuned_crv`.
- `tools/materialize_feature_prior_candidate.py` for the fastest highres
  candidate path when a candidate only needs feature-prior image scores and
  `feature_raw` pixel maps.

Current probes show that train-normal calibration helps structured categories
such as MVTec `grid` and `screw`, while tiny defects such as VisA `capsules`
and MPDD `bracket_white` need a feature prior rebuilt at 256px before
post-processing becomes meaningful. A fixed global `relu_robust_zscore +
highpass:9` policy was tested on all 33 categories and hurt the mean metrics,
so the main paper protocol remains `pixel_heatmap_source=feature_raw` until the
validation-selected or train-normal-gated policy has enough coverage to become
the paper-facing default.

The first held-out selector package is now available:

```powershell
python tools/export_heldout_pixel_policy_package.py --input mvtec15=tables\heldout_pixel_policy_mvtec15_highres --input visa=tables\heldout_pixel_policy_visa_highres --input mpdd=tables\heldout_pixel_policy_mpdd_highres --out tables\heldout_pixel_policy_package
```

It selects fixed robust-z + highpass9 for MVTec `grid`, selects `highres256`
for 13/15 MVTec classes, keeps MVTec `transistor` on `pixelraw`, and selects
`highres256` for all VisA and MPDD classes. The latest combined package
improves the mean held-out selected-vs-pixelraw deltas by `+0.1020` AUPRO,
`+0.0972` Pixel AP, and `+0.0937` Dice.

Highres candidate coverage is now broad enough that the next step is selector
stability and automatic gating, not simply adding more 256px runs. For any
additional candidate expansion, use the same feature-first runner with
`--image-size 256` and a distinct prefix, for example:

```powershell
python tools/run_feature_first_mvtec5.py --config configs/mvtec.yaml --categories grid,screw,zipper,capsule,pill,tile,transistor --run-prefix feature_highres_prior_mvtec15 --tables-out tables/feature_highres_prior_mvtec15 --image-size 256 --selector-only --skip-figures --resume
```

Selector stability across deterministic split seeds is summarized by:

```powershell
python tools/summarize_heldout_seed_packages.py --package 7=tables\heldout_pixel_policy_package --package 13=tables\heldout_pixel_policy_package_seed13 --package 23=tables\heldout_pixel_policy_package_seed23 --out tables\heldout_pixel_policy_seed_stability
```

The 3-seed stability table reports mean selected-vs-pixelraw deltas of
`+0.0935` AUPRO, `+0.0944` Pixel AP, and `+0.0938` Dice, with AP/Dice standard
deviation around `0.002-0.003`.

The current automatic gate probe is:

```powershell
python tools/select_pixel_policy_with_normal_gate.py --dataset mvtec15 --categories bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper --candidate 'pixelraw=runs/feature_pixelraw_mvtec15_{category}_feature_tuned_crv' --candidate 'fixed=runs/feature_fixedpixel_mvtec15_{category}_feature_pixel_policy' --candidate 'highres256=runs/feature_highres_prior_mvtec15_{category}_feature_tuned_crv' --out tables/normal_gate_highres_if_available_mvtec15 --val-ratio 0.5 --seed 7 --gate-metric highres_if_available
```

Repeating the same gate on VisA and MPDD and exporting the package to
`tables/normal_gate_highres_if_available_package` gives label-free
selected-vs-pixelraw deltas of `+0.0978` AUPRO, `+0.0931` Pixel AP, and
`+0.0883` Dice. A pure normal-rank proxy is much weaker (`+0.0380` Pixel AP,
`+0.0385` Dice), so the practical gate should default to highres and focus on
detecting exceptions such as MVTec `grid` and `transistor`.

The production paper selector replaces that heuristic with
`synthetic_normal_utility`. Every candidate must provide
`synthetic_validation.npz` and `synthetic_validation_metrics.json`; selection
uses synthetic localization quality, normal false positives, augmentation
stability, and latency. The validation-mask selector is an oracle upper bound
only.

Held-out SOTA comparison is exported with:

```powershell
python tools/export_heldout_sota_comparison.py --heldout mvtec15=tables\heldout_pixel_policy_mvtec15_post_padim --heldout visa=tables\heldout_pixel_policy_visa_post --heldout mpdd=tables\heldout_pixel_policy_mpdd_post --baseline-table mvtec15=tables\mvtec15_baselines\table_main_mvtec15.csv --baseline-table visa=tables\visa_baselines\table_main_visa.csv --baseline-table mpdd=tables\mpdd_baselines\table_main_mpdd.csv --out tables\heldout_sota_comparison_post_padim
```

This same-path held-out comparison currently gives category-level wins of
`27/33` for Pixel AP and `27/33` for Dice against the best local baseline per
category, with mean deltas of `+0.0621` Pixel AP and `+0.0584` Dice. The
current selected post/PadIM package is
`tables/heldout_pixel_policy_post_padim_package`.

For pure feature-prior highres candidates, the faster path is:

```powershell
python train_feature_prior.py --config configs/visa.yaml --category pcb3 --image-size 256 --max-samples 64 --run-name feature_highres_prior_visa_pcb3_models
python tools/materialize_feature_prior_candidate.py --config configs/visa.yaml --category pcb3 --feature-prior-checkpoint runs\feature_highres_prior_visa_pcb3_models\feature_prior.pt --image-size 256 --max-samples 64 --run-name feature_highres_prior_visa_pcb3_feature_tuned_crv --image-score-source feature_raw_cosine --image-score-mode top5 --pixel-heatmap-source feature_raw
```

After any highres expansion, rerun the selector with the expanded candidate
pool. Missing highres candidate runs are skipped, so this can be done
incrementally:

```powershell
python tools/select_pixel_policy_with_val_split.py --dataset mvtec15 --categories bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper --candidate 'pixelraw=runs/feature_pixelraw_mvtec15_{category}_feature_tuned_crv' --candidate 'fixed=runs/feature_fixedpixel_mvtec15_{category}_feature_pixel_policy' --candidate 'highres256=runs/feature_highres_prior_mvtec15_{category}_feature_tuned_crv' --out tables/heldout_pixel_policy_mvtec15_highres --val-ratio 0.5 --seed 7 --select-metric quality --materialize
python tools/select_pixel_policy_with_val_split.py --dataset visa --categories candle,capsules,cashew,chewinggum,fryum,macaroni1,macaroni2,pcb1,pcb2,pcb3,pcb4,pipe_fryum --candidate 'pixelraw=runs/feature_pixelraw_visa_{category}_feature_tuned_crv' --candidate 'fixed=runs/feature_fixedpixel_visa_{category}_feature_pixel_policy' --candidate 'highres256=runs/feature_highres_prior_visa_{category}_feature_tuned_crv' --out tables/heldout_pixel_policy_visa_highres --val-ratio 0.5 --seed 7 --select-metric quality --materialize
python tools/select_pixel_policy_with_val_split.py --dataset mpdd --categories bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes --candidate 'pixelraw=runs/feature_pixelraw_mpdd_{category}_feature_tuned_crv' --candidate 'fixed=runs/feature_fixedpixel_mpdd_{category}_feature_pixel_policy' --candidate 'highres256=runs/feature_highres_prior_mpdd_{category}_feature_tuned_crv' --out tables/heldout_pixel_policy_mpdd_highres --val-ratio 0.5 --seed 7 --select-metric quality --materialize
```

The same runner now supports cross-dataset expansion:

```powershell
python tools/run_feature_first_mvtec5.py --config configs/visa.yaml --categories auto --run-prefix feature_visa --tables-out tables/feature_visa --resume
python tools/run_feature_first_mvtec5.py --config configs/mpdd.yaml --categories auto --run-prefix feature_mpdd --tables-out tables/feature_mpdd --resume
```

Smoke runs for both configs have passed and produce dataset-specific aliases
`table_main_visa.csv` and `table_main_mpdd.csv`. Full feature-first VisA and
MPDD runs have also completed under `tables/feature_visa/` and
`tables/feature_mpdd/`.

Use the feature-first package exporter for paper-facing cross-dataset tables:

```powershell
python tools/export_feature_first_package.py --out tables/feature_paper_package
```

## Next-Phase Runner

`tools/run_next_phase_mvtec5.py` is the legacy diffusion-first evidence loop.
It:

- keeps new runs under `next_mvtec5_*`
- sweeps `reconstruction_steps=1,5,10`
- compares `residual_only`, `no_sev`, `synthetic_only_sev`, `no_prototype`, and full HN-SEV
- compares `no_crv`, `repair_visualization_only`, and full CRV fusion
- compares `fixed10`, `fixed25`, `rule_brds`, and `learned_lc_rds`
- searches CRV weights `0.25,0.35,0.5,1.0`
- writes `gate_summary.json` for the diffusion-first gate

## Required Ablations

- `residual only`
- `synthetic-only SEV`
- `HN-SEV without prototype`
- `HN-SEV with prototype`
- `without CRV`
- `fixed-10`, `fixed-25`, `rule BRDS`, `LC-RDS`
- `fixed resolution` vs `low-res global + native ROI`

## Current Prototype Evidence Status

Prototype bank training now uses calibrated novelty features from normal
patches plus mined hard-negative normal ROI patches. HN-SEV checkpoints save an
`operating_threshold`, FPRR is computed on normal images only, and the prototype
novelty score now conservatively modulates the final heatmap after CRV fusion.
The refreshed `next_mvtec5` evidence table reports prototype passes on
`bottle`, `cable`, and `zipper`; with HN-SEV, CRV, and LC-RDS also passing
5/5, `tables/next_mvtec5/gate_summary.json` is ready for MVTec15 expansion.

## Output Contract

Every run should produce:

- `metrics.json`
- `scores.csv`
- `predictions.npz`
- `heatmaps/`
- `masks/`
- `repairs/`
- `roi_budget.jsonl`
- `pareto.csv`
- `images/<case>/residual.png`
- `images/<case>/ground_truth.png`
- `images/<case>/final_heatmap.png`
- `images/<case>/pixel_eval_heatmap.png`
- `images/<case>/final_mask.png`

For feature-first runs, `predictions.npz` stores `heatmaps` as the selected
pixel evaluation map, `final_heatmaps` as the verifier/CRV-fused map, and
`score_heatmaps` as the image-score map.

The next-phase runner also writes:

- `table_reconstruction_sweep.csv`
- `table_ablation_hn_sev.csv`
- `table_ablation_crv.csv`
- `table_ablation_lc_rds.csv`
- `table_crv_weight_search.csv`
- `gate_summary.json`

## Current Completion Audit

Run `python tools/audit_lite_seer_plan.py` to verify the full plan against the
current artifacts. The current audit passes the MVTec5 gate, MVTec15 ours gate,
module ablation tables, metric contracts, reconstruction sweep, qualitative
case outputs, and repair-aware positioning checks. It is intentionally still
incomplete if `tools/audit_lite_seer_plan.py` reports missing baseline coverage.
The MVTec15 baseline table now has local runners available for PatchCore,
PaDiM, SimpleNet, DRAEM, RD4AD, UniAD, DiffusionAD, and DDAD.

## New Branch Audit

The weak-category multiscale branch is implemented by
`tools/materialize_multiscale_candidates.py`. On the aligned 64-image protocol,
the seed-7 label-free gate selected `post_highres_gaussian3` for four weak
categories and `padim128r18l123` for `transistor`; no multiscale candidate was
selected. This branch is retained as a negative ablation.

Retrieval-conditioned repair is implemented but disabled by default. The
fixed-10 `transistor` comparison in `tables/retrieval_ablation_transistor/`
shows lower Pixel AP/Dice for raw patch injection and spatially constrained
texture injection than no retrieval. It must not be presented as a positive
contribution unless later cross-category label-free evidence reverses this
result.
