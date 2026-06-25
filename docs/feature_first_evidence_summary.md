# Feature-First Lite-SEER-AD Evidence Summary

> Historical snapshot. The authoritative current results are in
> `tables/feature_first_fusion_aggregate_paper_package/` and
> `docs/results_limitations_draft.md`.

## Current Gate

The feature-first architecture flip has completed on MVTec5 and MVTec15 with
ImageNet-pretrained feature backbones.

- MVTec5 tables: `tables/feature_mvtec5/`
- MVTec15 tables: `tables/feature_mvtec15/`
- MVTec15 runs: `runs/feature_mvtec15_*`
- MVTec15 gate: `ready_for_mvtec15=true`
- MVTec15 categories: all 15 MVTec AD classes

Current MVTec15 gate:

| module | passes | total | ready |
|---|---:|---:|---|
| HN-SEV | 13 | 15 | true |
| CRV | 13 | 15 | true |
| LC-RDS | 14 | 15 | true |

The prototype branch is not required in the feature-first protocol because the
frozen feature prior supplies the main novelty map.

## Main Reading

Mean MVTec15 metrics:

| setting | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Dice | FPRR |
|---|---:|---:|---:|---:|---:|---:|
| residual_only | 0.5083 | 0.5356 | 0.1879 | 0.0199 | 0.0389 | 1.0000 |
| feature_only | 0.2786 | 0.9317 | 0.7457 | 0.2261 | 0.3015 | 1.0000 |
| feature_hn_sev | 0.2677 | 0.9248 | 0.7424 | 0.2002 | 0.2762 | 0.1268 |
| feature_hn_sev_crv | 0.3127 | 0.9209 | 0.7431 | 0.1935 | 0.2655 | 0.1298 |
| feature_tuned_crv, normalized top1 | 0.5768 | 0.9325 | 0.7580 | 0.2206 | 0.3029 | 0.1298 |
| feature_tuned_crv, raw distance top1 | 0.8597 | 0.9325 | 0.7580 | 0.2206 | 0.3029 | 0.1298 |
| feature_tuned_crv, raw cosine top5 image score | 0.8851 | 0.9325 | 0.7580 | 0.2206 | 0.3029 | 0.1298 |
| feature_tuned_crv, raw cosine top5 image score + raw feature pixel heatmap | 0.8851 | 0.9576 | 0.8282 | 0.3550 | 0.4173 | 0.1298 |

Interpretation:

- Feature-first detection fixes the weak RGB residual localization base by a
  large margin.
- HN-SEV reduces false-positive region rate from `1.0000` to roughly `0.13`,
  with a moderate localization tradeoff.
- Fixed `crv_weight=0.35` is too brittle; tuned CRV recovers Pixel AUROC,
  AUPRO, Pixel AP, and Dice while preserving FPRR.
- Raw feature magnitude is essential for image-level ranking. The current best
  fixed global scoring protocol is `feature_raw_cosine + top5`, which raises
  mean MVTec15 image AUROC to `0.8851`.
- Pixel localization should not be tied to the verifier-fused final map. A
  fixed `pixel_heatmap_source=feature_raw` raises MVTec15 Pixel AUROC from
  `0.9325` to `0.9576`, AUPRO from `0.7580` to `0.8282`, Pixel AP from
  `0.2206` to `0.3550`, and Dice from `0.3029` to `0.4173`.

## Raw Feature Image Scoring

The normalized feature heatmap is useful for ROI proposal, but weak for
image-level ranking because it removes absolute anomaly magnitude per image.
The pipeline now keeps raw PatchCore distance maps and cosine-gap maps for
image scoring, and explicitly decouples pixel localization through
`pixel_heatmap_source`.

Completed outputs:

- Runs: `runs/feature_rawscore_mvtec5_*`
- Tables: `tables/feature_rawscore_mvtec5/`
- Full MVTec15 raw-distance runs: `runs/feature_rawscore_mvtec15_*`
- Full MVTec15 raw-distance tables: `tables/feature_rawscore_mvtec15/`
- Full MVTec15 raw-cosine top5 runs:
  `runs/feature_cosine_top5_mvtec15_*`
- Full MVTec15 raw-cosine top5 tables:
  `tables/feature_cosine_top5_mvtec15/`
- Pixel-decoupled MVTec15 runs:
  `runs/feature_pixelraw_mvtec15_*`
- Pixel-decoupled MVTec15 tables:
  `tables/feature_pixelraw_mvtec15/`

Mean MVTec15 image-score readings:

| protocol | Image AUROC | note |
|---|---:|---|
| normalized final/feature maps | 0.5768 | weak image ranking |
| `feature_raw + top1` | 0.8597 | full MVTec15 complete |
| `feature_raw + p95` | 0.8806 | best raw-distance fixed global mode |
| `feature_raw_cosine + top5` | 0.8851 | current default fixed global mode |

The local MVTec15 baseline table reports PatchCore at `0.8854` image AUROC and
PaDiM at `0.9095`, so the latest Lite-SEER-AD protocol is now near PatchCore
but still not a SOTA claim.

Mean MVTec15 localization readings:

| protocol | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---:|---:|---:|---:|
| tuned final heatmap | 0.9325 | 0.7580 | 0.2206 | 0.3029 |
| `pixel_heatmap_source=feature_raw` | 0.9576 | 0.8282 | 0.3550 | 0.4173 |
| PatchCore local baseline | 0.9568 | 0.8323 | 0.4193 | 0.4787 |
| PaDiM local baseline | 0.9560 | 0.8314 | 0.4299 | 0.4911 |

Weak categories remain `grid`, `screw`, and `capsule`. Source/mode diagnostics
are stored under `tables/feature_rawscore_mvtec15/source_score_*`; per-category
source/mode selection reaches mean image AUROC `0.9144`, but that uses test
labels for model selection and should be treated only as an upper-bound
diagnostic.

## Weight And Score Search

CRV weight search:

- MVTec5: `tables/feature_mvtec5/table_crv_weight_search_feature.csv`
- MVTec15: `tables/feature_mvtec15/table_crv_weight_search_feature.csv`

Image-score search:

- MVTec15 normalized-map per-category best:
  `tables/feature_mvtec15/table_image_score_mode_search.csv`
- MVTec15 raw-distance top1:
  `tables/feature_rawscore_mvtec15/table_rawscore_mvtec15.csv`
- MVTec15 source/mode diagnostics:
  `tables/feature_rawscore_mvtec15/source_score_*`
- Current fixed global default: `feature_raw_cosine + top5`, mean image AUROC
  `0.8851`

Two additional tools were added for the next scoring pass:

- `tools/search_image_score_sources.py` searches fixed source/mode combinations
  against saved maps without retraining.
- `tools/materialize_fused_score.py` materializes fused image scores and
  supports `train_zscore` calibration from normal training samples.
- `tools/search_pixel_heatmap_sources.py` searches saved residual, feature,
  final, and score maps for localization without retraining.
- `tools/materialize_tuned_crv.py` now supports `--pixel-heatmap-source`, so
  image scoring, verifier-fused maps, and pixel evaluation maps are recorded as
  separate artifacts.

Initial train-normal fused scoring helps `grid` but hurts `capsule`, so it is
not yet the main protocol.

## Pixel AP / Dice Strengthening Probes

The current bottleneck after `pixel_heatmap_source=feature_raw` is no longer
Pixel AUROC/AUPRO alone; it is AP/Dice on small, thin, or very low-contrast
defects. Two additional diagnostic tools were added:

- `tools/materialize_fixed_pixel_policy.py` materializes a fixed train-normal
  calibration and post-processing policy without choosing modes from test
  labels.
- `tools/materialize_train_normal_pixel_calibration.py` builds train-normal
  spatial statistics and materializes calibrated pixel maps.
- `tools/search_pixel_postprocess.py` searches Gaussian smoothing, local
  high-pass contrast, top-hat, and closing operations from saved heatmaps.

Weak-category diagnostics:

| category | stage | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---|---:|---:|---:|---:|
| MVTec screw | raw feature | 0.9471 | 0.8198 | 0.0135 | 0.0316 |
| MVTec screw | train-normal z + highpass15 | 0.9610 | 0.8587 | 0.0339 | 0.0874 |
| MVTec grid | raw feature | 0.8624 | 0.5402 | 0.0576 | 0.1170 |
| MVTec grid | robust train-normal z + highpass9 | 0.9690 | 0.8945 | 0.2438 | 0.3491 |
| VisA capsules | raw feature, 128px | 0.8387 | 0.4641 | 0.0027 | 0.0134 |
| VisA capsules | 256px feature prior | 0.9712 | 0.8847 | 0.0208 | 0.0600 |
| MPDD bracket_white | raw feature, 128px | 0.9454 | 0.8103 | 0.0029 | 0.0079 |
| MPDD bracket_white | 256px feature prior | 0.9721 | 0.8520 | 0.0081 | 0.0234 |

Reading:

- Train-normal spatial calibration is promising for structured categories such
  as `grid` and `screw`.
- Simple post-processing is useful when applied after calibration, but is not
  sufficient by itself.
- Tiny/low-contrast defects such as VisA `capsules` and MPDD `bracket_white`
  benefit more from rebuilding the feature prior at 256px than from
  post-processing a 128px map.
- A fixed global policy, `relu_robust_zscore + highpass:9`, was evaluated on
  all 33 categories without test-label mode selection. It improves `grid`, but
  hurts most object categories, so it is not the main protocol.

Fixed global pixel policy vs the current `feature_raw` pixel protocol:

| dataset | protocol | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---|---:|---:|---:|---:|
| MVTec15 | `feature_raw` | 0.9576 | 0.8282 | 0.3550 | 0.4173 |
| MVTec15 | fixed robust-z + highpass9 | 0.8349 | 0.7220 | 0.1914 | 0.2782 |
| VisA | `feature_raw` | 0.9587 | 0.8033 | 0.1491 | 0.2204 |
| VisA | fixed robust-z + highpass9 | 0.8607 | 0.6266 | 0.0802 | 0.1461 |
| MPDD | `feature_raw` | 0.9525 | 0.8273 | 0.1803 | 0.2256 |
| MPDD | fixed robust-z + highpass9 | 0.7242 | 0.4691 | 0.0329 | 0.0674 |

The fixed-policy artifacts are stored under:

- `runs/feature_fixedpixel_*`
- `tables/feature_fixedpixel_mvtec15/`
- `tables/feature_fixedpixel_visa/`
- `tables/feature_fixedpixel_mpdd/`

These results mean the publishable strengthening path should not be a single
global post-processing rule. It should use a held-out validation split or a
stronger train-normal-only gate to decide whether to keep `feature_raw`, apply
structured-category calibration, or rebuild a 256px feature prior.

## Held-Out Pixel Policy Selection

A deterministic validation/test split selector was added to avoid choosing a
pixel policy from the same test subset used for reporting:

- `tools/select_pixel_policy_with_val_split.py`
- `tools/export_heldout_pixel_policy_package.py`

The selector chooses among saved candidate runs on a stratified validation
split, then reports the selected policy on the held-out split. The combined
package is stored under `tables/heldout_pixel_policy_package/`.

This selector uses real anomaly masks on its validation split and is now
classified as `oracle_upper_bound_only`. It is retained for diagnostic
headroom, not as the paper-facing method.

Candidate sets used in the current package:

| dataset | candidates | selected counts |
|---|---|---|
| MVTec15 | `pixelraw`, fixed robust-z + highpass9, `highres256` where available | `pixelraw`: 1, `fixed`: 1, `highres256`: 13 |
| VisA | `pixelraw`, fixed robust-z + highpass9, `highres256` where available | `highres256`: 12 |
| MPDD | `pixelraw`, fixed robust-z + highpass9, `highres256` where available | `highres256`: 6 |

Held-out means versus the `pixelraw` held-out baseline:

| dataset | method | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---|---:|---:|---:|---:|---:|
| MVTec15 | `pixelraw_heldout` | 0.8909 | 0.9575 | 0.8279 | 0.3740 | 0.4376 |
| MVTec15 | `selected_heldout` | 0.9144 | 0.9732 | 0.9274 | 0.4566 | 0.5065 |
| VisA | `pixelraw_heldout` | 0.9121 | 0.9545 | 0.7806 | 0.1697 | 0.2419 |
| VisA | `selected_heldout` | 0.9492 | 0.9863 | 0.8959 | 0.2871 | 0.3616 |
| MPDD | `pixelraw_heldout` | 0.7823 | 0.9492 | 0.8174 | 0.1883 | 0.2302 |
| MPDD | `selected_heldout` | 0.8738 | 0.9745 | 0.8992 | 0.2819 | 0.3337 |

Mean selected-vs-pixelraw held-out deltas across all selected categories:

| metric | mean delta |
|---|---:|
| Image AUROC | +0.0408 |
| Pixel AUROC | +0.0234 |
| AUPRO | +0.1020 |
| Pixel AP | +0.0972 |
| Dice | +0.0937 |

The same selector was also repeated on three deterministic split seeds
(`7`, `13`, and `23`) after highres coverage was expanded:

| metric | mean delta across seeds | std | min | max |
|---|---:|---:|---:|---:|
| Image AUROC | +0.0475 | 0.0047 | +0.0408 | +0.0512 |
| Pixel AUROC | +0.0223 | 0.0010 | +0.0209 | +0.0234 |
| AUPRO | +0.0935 | 0.0074 | +0.0840 | +0.1020 |
| Pixel AP | +0.0944 | 0.0022 | +0.0920 | +0.0972 |
| Dice | +0.0938 | 0.0028 | +0.0904 | +0.0973 |

Reading:

- The validation-selected policy is now a viable non-test-selected
  strengthening mechanism.
- The first selected policies improve held-out AUPRO, Pixel AP, and Dice on
  average. The MPDD, VisA, and MVTec15 highres expansions now all show that
  validation-selected resolution scaling can move AP/Dice by a material amount
  rather than a tiny diagnostic gain.
- The latest expansion adds the remaining VisA classes and the remaining
  MVTec fallback classes. The refreshed selector now chooses `highres256` for
  all VisA and MPDD classes, and for 13/15 MVTec classes.
- The next high-impact experiment is no longer simple highres coverage. It is
  a train-normal or held-out-normal gate for deciding when highres, raw feature
  maps, or structured calibration should be used, plus repeat-seed validation
  of the now-material AP/Dice gains. The first 3-seed stability pass shows that
  AP/Dice gains are stable across deterministic validation/test splits.

## Normal-Gated Pixel Policy Probes

The labeled validation selector is useful as a clean held-out estimate, but a
paper-facing deployment story still needs a policy that does not choose pixel
maps from anomaly masks. A normal-gate selector was added:

- `tools/select_pixel_policy_with_normal_gate.py`

It writes the same `candidate_split_metrics.csv`, `selection.csv`, and
`selected_heldout_metrics.csv` schema as the labeled selector, so the existing
package exporter can compare normal-gated policies directly.

Seed-7 policy comparison against the `pixelraw` held-out baseline:

| policy | selected counts | Image AUROC delta | Pixel AUROC delta | AUPRO delta | Pixel AP delta | Dice delta |
|---|---|---:|---:|---:|---:|---:|
| labeled validation selector | MVTec 13 highres + 1 fixed + 1 pixelraw; VisA/MPDD all highres | +0.0408 | +0.0234 | +0.1020 | +0.0972 | +0.0937 |
| automatic `highres_if_available` | all 33 categories highres | +0.0439 | +0.0216 | +0.0978 | +0.0931 | +0.0883 |
| normal-only rank-sum proxy | 14 highres + 19 pixelraw | +0.0206 | +0.0132 | +0.0486 | +0.0380 | +0.0385 |

Reading:

- A simple automatic highres policy already preserves most of the AP/Dice gain
  without using validation masks.
- Pure normal-noise minimization is too conservative: it rejects highres on
  many categories where highres materially improves anomaly localization.
- The next gate should keep highres as the default and learn only the exceptions
  (`grid` prefers structured calibration; `transistor` still prefers
  `pixelraw` under the labeled selector).

## Synthetic-Normal Utility Gate

The paper-facing automatic selector is now implemented as
`synthetic_normal_utility`.

- `tools/materialize_synthetic_normal_validation.py` rebuilds each candidate on
  held-out normal images and deterministic synthetic defects.
- `tools/select_pixel_policy_with_normal_gate.py` combines synthetic Pixel AP,
  AUPRO, Dice, Pixel/Image AUROC, normal false-positive rate, augmentation
  stability, and measured latency.
- Selection artifacts explicitly record
  `uses_real_anomaly_labels_for_selection=false` and
  `uses_real_anomaly_masks_for_selection=false`.
- `tools/run_synthetic_normal_gate.py` executes the protocol for seeds
  `7,13,23`.

The first end-to-end probe on MVTec `transistor` selected the ResNet18
multi-layer PaDiM candidate without real anomaly labels. On the held-out split
it reached Pixel AP `0.6117` and Dice `0.6136`, closing the largest remaining
category gap in the previous package.

## Held-Out SOTA Comparison

The strengthened held-out selector was also compared against local baseline
prediction files on exactly the same held-out sample paths:

- `tools/export_heldout_sota_comparison.py`
- Current package: `tables/heldout_sota_comparison_post_padim/`

The exporter aligns each selected Lite-SEER-AD held-out sample to baseline
`predictions.npz` files by path before recomputing metrics, so the comparison
does not mix different test subsets.

Mean metrics on the aligned held-out split:

| dataset | method | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---|---:|---:|---:|---:|---:|
| MVTec15 | best local baseline by mean AP/Dice: PaDiM | 0.9138 | 0.9668 | 0.8555 | 0.4088 | 0.4747 |
| MVTec15 | Lite-SEER-AD selected held-out | 0.9162 | 0.9757 | 0.9326 | 0.4672 | 0.5151 |
| VisA | best local baseline by mean AP/Dice: PatchCore | 0.8525 | 0.9585 | 0.7890 | 0.1796 | 0.2573 |
| VisA | Lite-SEER-AD selected held-out | 0.9492 | 0.9864 | 0.8964 | 0.2888 | 0.3634 |
| MPDD | best local baseline by mean AP/Dice: PatchCore | 0.7452 | 0.9506 | 0.8277 | 0.2188 | 0.2615 |
| MPDD | Lite-SEER-AD selected held-out | 0.8738 | 0.9681 | 0.8799 | 0.2857 | 0.3447 |

Category-level wins against the best local baseline per category:

| metric | wins / total | mean delta |
|---|---:|---:|
| Image AUROC | 24 / 33 | +0.0013 |
| Pixel AUROC | 25 / 33 | +0.0120 |
| AUPRO | 25 / 33 | +0.0581 |
| Pixel AP | 27 / 33 | +0.0621 |
| Dice | 27 / 33 | +0.0584 |

Remaining AP/Dice gaps are concentrated in a small set:

- Pixel AP losses: MVTec `cable`, `capsule`, `hazelnut`, `screw`,
  `transistor`; VisA `fryum`.
- Dice losses: MVTec `cable`, `capsule`, `hazelnut`, `screw`, `transistor`;
  VisA `fryum`.

## Cross-Dataset Status

The feature-first runner now supports `--categories auto` and writes
dataset-specific aliases such as `table_main_visa.csv` and
`table_main_mpdd.csv`.

Smoke checks have completed with random feature weights for plumbing only. New
smoke invocations write to separate `*_smoke` table directories to avoid
contaminating full CRV weight recommendations:

| dataset | category | report | table alias |
|---|---|---|---|
| VisA | candle | `runs/feature_visa_smoke_report.json` | `tables/feature_visa_smoke/table_main_visa.csv` |
| MPDD | bracket_black | `runs/feature_mpdd_smoke_report.json` | `tables/feature_mpdd_smoke/table_main_mpdd.csv` |

These smoke runs validate the latest `feature_raw_cosine + top5` protocol on
non-MVTec configs. They are not paper evidence.

Full paper-facing VisA and MPDD runs have now completed with ImageNet
pretrained features. The latest paper-facing package uses
`image_score_source=feature_raw_cosine`, `image_score_mode=top5`, and
`pixel_heatmap_source=feature_raw`:

| dataset | categories | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---|---:|---:|---:|---:|---:|---:|
| MVTec15 | 15 | 0.8851 | 0.9576 | 0.8282 | 0.3550 | 0.4173 |
| VisA | 12 | 0.9182 | 0.9587 | 0.8033 | 0.1491 | 0.2204 |
| MPDD | 6 | 0.7609 | 0.9525 | 0.8273 | 0.1803 | 0.2256 |

Feature-first paper package:

- `tables/feature_paper_package/table_main_cross_dataset.csv`
- `tables/feature_paper_package/table_mean_by_dataset_method.csv`
- `tables/feature_paper_package/table_category_deltas.csv`
- `tables/feature_paper_package/summary.json`

Pre-strengthening `feature_raw` cross-dataset comparison against the best local
baseline by category:

| metric | wins / total | mean delta |
|---|---:|---:|
| Image AUROC | 12 / 33 | -0.0461 |
| Pixel AUROC | 11 / 33 | -0.0095 |
| AUPRO | 13 / 33 | -0.0248 |
| Pixel AP | 7 / 33 | -0.0926 |
| Dice | 9 / 33 | -0.0875 |

Interpretation: the pixel-source decoupling materially improves localization
and makes Pixel AUROC/AUPRO competitive on many categories, but the
pre-strengthening `feature_raw` package is not enough for a SOTA claim. The
held-out highres selector above is the current paper-facing strengthening path.

## Next Move

The code path is now strong enough for paper-facing held-out SOTA comparison.
The next work should focus on closing the remaining AP/Dice losses and making
the automatic gate less heuristic:

1. Materialize `synthetic_normal_utility` evidence for the complete fixed
   candidate pool on all 33 categories and seeds `7,13,23`.
2. Close the remaining held-out gaps, starting with MVTec `hazelnut`,
   `capsule`, `screw`, and `cable`, then VisA `fryum`.
3. Improve weak-category image scoring for `grid`, `screw`, and `capsule`
   without test-label selection.
4. Add train-normal or held-out-normal validation for choosing among
   raw-distance, cosine-gap, and fused score streams.
5. Refresh qualitative failure analysis using `tables/feature_paper_package`
   and prioritize categories where `feature_raw` improves AUROC/AUPRO but
   still produces weak AP/Dice.
