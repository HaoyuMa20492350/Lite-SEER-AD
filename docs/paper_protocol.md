# Lite-SEER-AD Paper Protocol

## Main Method

- Architecture: frozen feature-first detection and localization, with
  HN-SEV providing verification filtering, CRV providing visualization and
  post-hoc inspection, and LC-RDS providing efficiency evidence.
- Pixel-policy protocol:
  `normal_plus_synthetic_cross_seed_mean_no_real_anomaly_labels`.
- Seeds: `7, 13, 23`.
- Datasets: MVTec AD 15 categories, VisA 12 categories, and MPDD 6
  categories.
- Candidate pool: `pixelraw`, `highres256`, fixed normal calibration,
  Gaussian post-processing, ResNet18 multi-layer PaDiM, predeclared
  multiscale/retrieval negative branches, and normal-calibrated
  highres/pixelraw plus PaDiM/highres heatmap fusion.
- Selection data: deterministic subsets of official training-normal images
  and deterministic blob/scratch synthetic defects. The full DTD texture
  bank is used by HN-SEV, not by the frozen main pixel-threshold protocol.
- Selection stabilization: aggregate synthetic-normal threshold evidence
  across seeds `7, 13, 23`, then freeze one threshold per category.
- Forbidden selection inputs: real anomaly labels and real anomaly masks.
- Pixel threshold:
  `synthetic_normal_fixed_threshold_v1`, selected from retained normal
  heatmaps and deterministic synthetic defects with normal-pixel FPR capped
  at `0.5%`. Among feasible thresholds, synthetic Dice is maximized.
- Main F1/IoU/Dice use the frozen threshold. Test-GT optimized values are
  stored only as `oracle_f1`, `oracle_iou`, `oracle_dice`, and
  `oracle_threshold`.
- AUPRO protocol: component-level PRO AUC over background FPR `[0, 0.3]`,
  duplicate-FPR collapse, and interpolation at the `0.3` boundary.

## Oracle

`select_pixel_policy_with_val_split.py` uses real anomaly masks. Its results are
an oracle upper bound for measuring headroom and must not be presented as the
main method.

## Required Evidence

Every selected run must contain:

- `selection_evidence.json`;
- explicit `detection_heatmaps`, `verification_heatmaps`, and
  `image_score_heatmaps` arrays in `predictions.npz` (legacy aliases are kept
  for compatibility);
- the three evidence seeds used for cross-seed aggregation;
- normal false-positive statistics;
- synthetic Pixel AP, AUPRO, Dice, and AUROC;
- augmentation stability;
- measured candidate latency;
- explicit false values for real-anomaly label and mask use.
- `pixel_threshold_policy.json` with the evidence seeds, threshold, normal
  FPR, and false real-anomaly-use flags.

The frozen workspace manifest is written by
`tools/freeze_evidence_protocol.py`.

The authoritative paper-facing package is
`tables/feature_first_fusion_aggregate_paper_package/`.
Independent module claims use the 33-category
`module_evidence_summary.json`, not the historical MPDD-only table.

## Optional Branches

- Retrieval-conditioned repair is opt-in via
  `infer.py --enable-retrieval-repair`; it is disabled in the main protocol
  because the label-free multi-category ablation is negative.
- Multiscale feature variants remain disabled because their formal ablation is
  negative.
- Normal-calibrated fusion is eligible because its calibration uses retained
  normal artifacts only and it enters through the same cross-seed
  synthetic-normal gate.
- Validation-mask policy selection remains an oracle upper bound and cannot
  choose the paper-facing method.

## Claim Boundary

- Supported: label-free candidate selection, three-dataset localization gains,
  cross-seed selection stability, HN-SEV verification filtering, CRV
  structural-fidelity visualization, and LC-RDS efficiency evidence.
- HN-SEV lowers FPRR in 33/33 categories. LC-RDS is consistently faster than
  fixed25 and rule-based repair in 33/33 categories, but is faster than
  fixed10 in only 16/33 categories.
- Unsupported: universal external SOTA.
- The pinned official PatchCore comparison is complete on MVTec15. It supports
  only a mean AUPRO delta of `+0.0106`; PatchCore remains stronger on mean
  Image AUROC, Pixel AUROC, Pixel AP, and fixed Dice.
- Local baseline runners are engineering controls and must be displayed as
  `PatchCore-Local`, `PaDiM-Local`, or `*-Lite`; they are not official
  reproductions.
- Unsupported in the frozen configuration: general Image AUROC improvement or
  a claim that CRV improves the detector's Pixel AP/Dice, performs semantic
  defect removal, or is positively aligned with real defect masks. The pooled
  33-category SDR-GT Spearman correlation is `-0.1235`.
- Comparable latency uses `synchronized_batch_latency_v1`: batch size 1,
  50 warmups, 200 measurements, and CUDA synchronization. Legacy rows without
  this field are not valid for cross-method efficiency claims.
- Lite-SEER-AD inference additionally writes `component_latency.csv` with
  synchronized detector, HN-SEV, repair, and end-to-end per-image timing.
  This decomposition is reported separately from the repeated comparable
  benchmark.

## Optional Supplementary Assets

MVTec AD 2 is not part of the main paper protocol and does not gate
submission. Its completed local public/private pipeline and checker-passed
archive are retained only for a possible future appendix:
`submissions/mvtec_ad2_seed7_model256.tar.gz`, SHA256
`9ae82e41018ad17a59c5f9ae176f4a12355cef9f7bc9bc143e342145deff266b`.
No private metric or leaderboard claim is permitted without a verifiable
official server result.
