# Lite-SEER-AD Results and Limitations Draft

## Experimental Protocol

The paper-facing method uses a frozen feature-first detector and separates its
localization heatmap from repair-based verification. Candidate selection uses
retained normal images, deterministic synthetic defects, augmentation
stability, normal-pixel false-positive rate, and measured latency. It never
uses real anomaly labels or masks.

Pixel thresholds are frozen with
`synthetic_normal_fixed_threshold_v1`. Normal-pixel FPR is capped at `0.5%`;
among feasible thresholds, synthetic Dice is maximized. Main F1/IoU/Dice use
this fixed threshold. Test-GT optimized values are retained only as
`oracle_*` diagnostics.

The three synthetic-evidence seeds `7/13/23` are averaged to freeze one
candidate per category. That candidate is then evaluated on three independent
held-out splits. The pool contains pixelraw, high-resolution, Gaussian
post-processing, structured calibration, PaDiM, negative multiscale
ablations, and normal-calibrated fusion candidates.

## Main Results

Across MVTec15, VisA, and MPDD, the protocol covers 33 categories and all nine
dataset-seed runs. Candidate selection agreement is `100%`.

Against the strongest path-aligned local engineering baseline for each
category, the threshold-independent metrics are:

| Metric | Mean delta | 95% bootstrap CI | Wins |
|---|---:|---:|---:|
| AUPRO | +0.0584 | [+0.0320, +0.0918] | 30/33 |
| Pixel AP | +0.0618 | [+0.0419, +0.0821] | 31/33 |

The category-level sign tests are also positive overall: `p=1.40e-6` for
AUPRO and `p=1.31e-7` for Pixel AP. Image AUROC is not improved reliably: its
overall confidence interval crosses zero.

The local runners are displayed as `PatchCore-Local`, `PaDiM-Local`, and
`*-Lite`. They are not official reproductions, so these results support a
controlled engineering comparison rather than universal external SOTA.

Seven external baselines are complete for all 15 MVTec categories. Every row
uses the same metric implementation and
`synthetic_normal_fixed_threshold_v1` without real anomaly labels or masks.
PaDiM is a maintained Anomalib reference; the other six are pinned
author-official implementations.

| Method | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Fixed Dice |
|---|---:|---:|---:|---:|---:|
| Lite-SEER-AD | 0.9278 | 0.9785 | 0.9220 | 0.4639 | 0.4778 |
| PatchCore-Official | 0.9893 | 0.9821 | 0.9114 | 0.6236 | 0.4896 |
| PaDiM-Anomalib | 0.8903 | 0.9596 | 0.8797 | 0.4001 | 0.3731 |
| UniAD-Official | 0.9764 | 0.9703 | 0.9045 | 0.4481 | 0.4332 |
| DRAEM-Official | 0.9805 | 0.9749 | 0.9279 | 0.6886 | 0.5232 |
| DDAD-Official | 0.9888 | 0.9748 | 0.9085 | 0.6325 | 0.3420 |
| RD4AD-Official | 0.9789 | 0.9780 | 0.9339 | 0.5680 | 0.4901 |
| SimpleNet-Official | 0.9900 | 0.9752 | 0.9049 | 0.5705 | 0.3528 |

Lite-SEER-AD exceeds the PaDiM reference on all five means and exceeds UniAD
on four localization means, but DRAEM and RD4AD have higher mean AUPRO.
Most author-official methods also have stronger Image AUROC and Pixel AP.
The result supports the label-free policy and threshold contribution, not
overall superiority or SOTA.

Paired inference over the 15 matched categories narrows the claim further.
Against PaDiM, Pixel AUROC improves by `+0.0189` with 95% bootstrap CI
`[+0.0106, +0.0280]` and Holm-adjusted exact sign-test `p=0.0068`.
PaDiM AUPRO, Pixel AP, and fixed-Dice intervals are also positive, but their
Holm-adjusted sign tests are not significant. The positive AUPRO point
estimates against PatchCore, UniAD, DDAD, and SimpleNet all have intervals
crossing zero. DRAEM and DDAD retain statistically supported Pixel AP
advantages after Holm correction.

## Image-Score Aggregation Audit

We predeclared 11 image-score aggregators spanning maximum, mean, percentile,
and top-fraction pooling. One mode per MVTec category was selected using only
retained normal images and deterministic synthetic defects at seeds
`7/13/23`, then frozen and audited over 45 held-out runs.

The selected modes achieve mean Image AUROC `0.9263`, compared with `0.9278`
for the existing global `top5` rule (`-0.0015`). Four categories improve,
four degrade, and seven are unchanged. The largest gain is `pill +0.0378`;
the largest loss is `screw -0.0586`. We therefore retain `top5`. No switching
threshold is tuned after this audit because doing so would use held-out labels
at the meta-selection level.

The fixed-threshold absolute results over 99 held-out runs are:

| Dataset | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Fixed Dice | Oracle Dice |
|---|---:|---:|---:|---:|---:|---:|
| MVTec15 | 0.9278 | 0.9785 | 0.9220 | 0.4639 | 0.4778 | 0.5159 |
| VisA | 0.9609 | 0.9865 | 0.8986 | 0.2600 | 0.2222 | 0.3391 |
| MPDD | 0.8198 | 0.9744 | 0.9069 | 0.2762 | 0.2663 | 0.3257 |

The maximum observed threshold-evidence normal FPR is `0.4968%`. Fixed Dice
is `0.0706` lower than oracle Dice on average, quantifying the optimism of
test-GT threshold selection. The historical oracle Dice baseline deltas are
therefore excluded from the main claim.

## Stability And Label-Free Selection

Relative to the same-path pixelraw candidate, the threshold-independent mean
gains are `+0.0251` Pixel AUROC, `+0.0977` AUPRO, and `+0.1033` Pixel AP.
The historical Dice selection statistic remains useful for diagnosing
candidate stability but is not a paper-facing detector comparison.

All 99 selection records explicitly store false values for real-anomaly label
and mask use. Each record also identifies synthetic evidence seeds
`7 13 23`.

## Fusion Candidate Contribution

Normal-calibrated fusion estimates a fixed center and scale from retained
normal heatmaps. It therefore preserves cross-image anomaly magnitude while
allowing candidates with different resolutions and score scales to be
combined.

- Highres/pixelraw `70/30` is selected in 22 of 33 categories.
- PaDiM/highres `50/50` is selected in three of the five MVTec categories
  where the PaDiM source is available.
- The fusion branches are selected only through the same synthetic-normal
  utility and contain no category-name rules.

This addition raises the main comparison from the earlier AP/Dice
`25/33`/`23/33` result to `31/33`/`27/33`.

## Module Evidence

The frozen detector metrics are intentionally unchanged by HN-SEV, CRV, and
LC-RDS. Their independent 33-category evidence is reported separately:

- HN-SEV reduces mean FPRR by `0.9187`, with lower FPRR in all `33/33`
  categories.
- The ROI-mask audit shows this filtering is aggressive: over 5,091 resolved
  ROIs, background/normal ROI suppression is `93.01%`, but GT-overlapping ROI
  retention is only `15.92%` and image-level ROI recall drops from `87.49%`
  before HN-SEV to `15.12%` after HN-SEV. HN-SEV should therefore be claimed
  only as a false-positive suppression module. The coverage audit in
  `tables/hn_sev_input_ablation/` now contains exact synthetic-only,
  +clean-normal, +hard-negative, and feature/prototype variants with metric rows
  for all 33 categories; these ablations complete the evidence trail but do not
  recover recall safety.
- The earlier MPDD-only CRV ablation reports positive aggregate RDC/SDR, but
  the complete cross-dataset audit below does not support GT-aligned repair.
  CRV is therefore restricted to visualization and post-hoc inspection.
- LC-RDS reduces mean latency by `95.04` ms versus fixed25 and by `85.05` ms
  versus rule-based repair, with lower latency in `33/33` categories for both
  comparisons. Versus fixed10, the pooled reduction is only `0.51` ms and
  only `16/33` categories are faster, so no universal fastest-policy claim is
  made.
- Retrieval-conditioned repair and the tested multiscale branches remain
  negative ablations and are disabled by default.

## Cross-Dataset Repair Audit

The independent 128-pixel HN-SEV/CRV module runs cover all 33 categories,
2,066 images, and 5,091 ROIs. These runs are separate from the high-resolution
main detector and must not be used as detector AP/Dice evidence.

On 1,111 anomalous images, repair preserves local structure with mean SSIM
`0.9229`, background PSNR `29.18` dB, background MAE `0.00827`, and boundary
consistency `0.9173`. These are structural-fidelity and edit-locality
measurements. Because no paired anomaly-free target exists, they do not prove
semantic defect removal.

SDR is not positively aligned with real defect occupancy. Across 2,246
anomaly-image ROIs, pooled SDR-GT Spearman correlation is `-0.1235`; the
pixel, feature, and prototype-space values are `-0.1527`, `-0.0192`, and
`+0.0007`. Positive-SDR ROIs hit the real mask `44.97%` of the time, below
the overall anomaly-ROI hit rate of `55.92%`. Consequently, the paper makes
no GT-aligned repair, verification-accuracy, or detector-gain claim for CRV.

## Repair Executor Necessity Audit

A clean-target synthetic-defect audit now evaluates non-diffusion repair
executors on 33 normal images, one per category across MVTec15, VisA, and
MPDD. The original normal image is the target, so PSNR, SSIM, foreground MAE,
background MAE, and latency are directly measurable. The tested non-diffusion
executors are corrupted identity, mean fill, neighbor-mean inpainting,
partial-conv proxy inpainting, trained linear partial-conv inpainting, nearest
normal patch, blur inpainting, a downsample/upsample light-AE proxy, a trained
PCA light-AE, and a multiscale light-U-Net proxy.

The strongest quality non-diffusion baseline is `partial_conv_inpaint`, with
mean PSNR `37.12`, mean SSIM `0.9826`, mean LPIPS `0.0191`, foreground MAE
`0.0696`, and zero background MAE under this local-mask protocol.
The faster `neighbor_mean_inpaint` still reaches mean PSNR `34.43`, mean SSIM
`0.9684`, and mean LPIPS `0.0362`. The same-protocol category diffusion
executor is now evaluated over the same 33 images, but it reaches only mean
PSNR `21.73`, mean SSIM `0.9378`, and mean LPIPS `0.1319`, with mean latency
`67.94` ms. This answers the necessity question negatively: diffusion is not
Pareto-preferred under this clean-target repair protocol. The trained
`trained_pca_light_ae` baseline is faster than the
partial-conv proxy but weaker on the quality metrics, with mean PSNR `31.84`,
mean SSIM `0.9674`, and mean LPIPS `0.0481`.
The trained `trained_linear_partial_conv` baseline reaches mean PSNR `33.44`,
mean SSIM `0.9719`, and mean LPIPS `0.0373`; it is useful evidence for a
learned non-diffusion inpainting alternative but still does not dominate the
stronger iterative partial-conv proxy. Because wall-clock latency varies by
run, the paper should cite latency from
`tables/repair_executor_ablation/table_repair_executor_summary.csv` generated
with the submitted artifact bundle.
The paper should therefore treat diffusion as a replaceable repair executor,
not as a necessary component. The release gate for this audit is passed because
same-protocol diffusion, LPIPS, non-diffusion families, and the Pareto decision
are all present in `tables/repair_executor_ablation/`.

## Deployment Latency Audit

A synchronized batch-1 deployment smoke table is available in
`tables/deployment_latency/`. It reports separate IO, detector, verifier,
scheduler, repair, and end-to-end proxy timings under
`synchronized_batch_latency_v1`, including p50, p95, and p99 latency. On the
current RTX 4090 Laptop GPU environment, the smoke end-to-end proxy has mean
latency `0.3600` ms, p95 `0.4480` ms, and p99 `0.4737` ms. The table also
records hardware metadata, peak allocated GPU memory for the smoke benchmark,
and the current LC-RDS offline budget-violation source.

The LC-RDS action-space audit in
`tables/lc_rds_budget_audit/table_action_space.csv` separates required,
configured, implemented, and observed actions. The v2 config and scheduler
code now cover `skip`, `repair-5`, `repair-10`, `repair-25`, and
`native-refine`. The historical ROI logs only observe `repair-10`, so they
are retained as offline replay evidence rather than production budget proof.
`tables/lc_rds_budget_sweep/` adds a measured synthetic action sweep: all five
required actions have batch-1 wall-clock latency samples, and the six paper
budgets report synthetic accounting budget violation rate `0.0`. The same
export also replays measured action p95 latency on the 33-category ROI logs
covering 2066 images and 5091 ROI rows, with ROI measured budget violation
rate `0.0`. This closes the action-executability and measured-latency replay
gap while keeping historical ROI replay separate from production measurement.

`tables/deployment_production_latency/` now summarizes real `utility_lc_rds`
source inference component logs for all 33 categories plus the executed
`production_lc_rds_budget_guarded*` runs. The current summary records `231`
runs and `30415` images. This upgrades the evidence from smoke-only component
proxies to audited production inference timings for the detector, HN-SEV
verifier, repair stage, and end-to-end path.

This is still not a final cross-hardware deployment-speed claim. The production
budget matrix now covers all 33 dataset/category pairs under all six budgets
(`10/25/50/75/100/150 ms`), so `budget_runs=198/198` and
`missing_budget_runs=0`. The guarded production sweep observes all five
required actions and has maximum budget violation rate `0.0009206`, below the
1% gate. A single-hardware production-style energy probe is also recorded in
`tables/deployment_production_latency/energy_measurements/`, with `274.41 J`
over an 8-image `mvtec15/bottle` budget-10 run. Cross-hardware validation is
still required before making deployment-speed claims beyond the current
hardware profile.

## Few-Shot Results

MVTec15 few-shot evaluation contains 135 completed runs:
15 categories, shots `8/16/32`, and seeds `7/13/23`.

| Shots | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Dice |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.8825 | 0.9626 | 0.8835 | 0.3659 | 0.4276 |
| 16 | 0.8929 | 0.9674 | 0.8974 | 0.3979 | 0.4579 |
| 32 | 0.9113 | 0.9718 | 0.9100 | 0.4213 | 0.4787 |

All localization metrics improve monotonically with the number of normal
training images.

## Optional Supplementary Asset: MVTec AD 2

The local AD2 pipeline is complete: 24/24 public runs, 4090 private and
private-mixed images, official checker pass, and archive
`submissions/mvtec_ad2_seed7_model256.tar.gz`. The 256-pixel archive is
`341.67 MiB`, contains 8180 files, and has SHA256
`9ae82e41018ad17a59c5f9ae176f4a12355cef9f7bc9bc143e342145deff266b`.
Public threshold-independent means are
Image AUROC `0.7127`, Pixel AUROC `0.8149`, AUPRO `0.4712`, and Pixel AP
`0.1001`.

MVTec AD 2 is excluded from the main paper protocol because authenticated
server evaluation is unavailable. These local artifacts are retained only as
an optional appendix asset and do not gate submission or support a private
benchmark claim.

## Remaining Failure Cases

The two Pixel AP losses are MVTec `hazelnut` and `capsule`. AUPRO losses
remain on VisA `chewinggum` and MVTec `screw` and `transistor`.

These cases show that synthetic defect masks do not perfectly reproduce real
defect area, topology, or threshold behavior. They should be included as
failure examples rather than hidden by aggregate results.

The frozen seed-7 post-hoc panel is
`paper/figures/fig_frozen_failure_cases.png`, with its auditable case index in
`tables/failure_case_panel_mvtec15/`. It shows complete fixed-threshold misses
for the selected `grid` and `pill` examples and over-segmentation for
`hazelnut`. Real masks are used only for this diagnostic case selection.

The broader weak-category audit is now recorded in
`tables/failure_taxonomy/`. It covers the retained weak categories
(`grid`, `screw`, `pill`, `capsule`, `hazelnut`, `transistor`, VisA
`capsules`, VisA `fryum`, and MPDD `bracket_white`) as post-hoc failure
taxonomy only. The taxonomy uses real masks for diagnosis, not for method,
candidate, or threshold selection.

## Limitations

1. Seven external baselines are complete (`105/120` matrix entries). The
   remaining 15 entries are DiffusionAD, whose author configuration requires
   3000 epochs per category and exactly 663,000 optimizer steps under
   `batch size 16` with `drop_last=True`. Source, assets, strict runner, and
   smoke validation are complete, but partial-training checkpoints and
   predictions are not retained or eligible for paper comparison. Measured
   16-GB throughput projects to about 3,801 aggregate GPU hours.
2. Cross-seed evidence aggregation improves selection stability but uses three
   deterministic synthetic validation seeds. Generalization to other
   synthesis families remains untested. The image-score aggregation audit
   additionally shows that synthetic-domain ranking does not reliably predict
   real-defect ranking.
3. Historical Pareto rows mix latency protocols. Only new rows carrying
   `synchronized_batch_latency_v1` are valid for cross-method comparison.
4. HN-SEV currently has a recall-safety limitation. The new audit in
   `tables/hn_sev_retention_calibration/` resolves all 5,091 ROI rows against
   dataset masks and finds strong background suppression but substantial
   TP-retention loss. The companion coverage audit in
   `tables/hn_sev_input_ablation/` now completes the exact synthetic-only,
   +clean-normal, +hard-negative, and +feature/prototype input ablations for all
   33 categories. Because the recall loss remains, HN-SEV must still not be
   described as recall-safe.
5. Diffusion repair necessity is answered negatively. The clean-target audit in
   `tables/repair_executor_ablation/` covers simple inpainting, partial-conv
   proxy, trained linear partial-conv, nearest normal patch, trained PCA
   light-AE, light-AE proxy, light-U-Net proxy, same-protocol category
   diffusion, and real LPIPS. The Pareto decision is `diffusion_not_necessary`,
   so diffusion must not be written as the preferred or necessary executor.
6. Deployment latency now has both a synchronized component smoke audit and a
   production inference component summary over 33 `utility_lc_rds` source runs
   plus the full 198-run guarded production LC-RDS budget sweep. LC-RDS also
   has a five-action config/code coverage audit, a measured synthetic action
   sweep, a measured-latency ROI budget replay, and a single-hardware
   production-style energy probe. The remaining deployment blocker is
   cross-hardware evidence.
7. CRV does not show positive pooled SDR-GT alignment. It is retained only
   for visualization and post-hoc inspection, and does not support semantic
   repair, verification-accuracy, AP, or Dice claims.

## Evidence Files

- Main package: `tables/feature_first_fusion_aggregate_paper_package/`
- Acceptance matrix: `tables/feature_first_plan_acceptance/`
- Stability: `tables/synthetic_gate_fusion_aggregate_stability/`
- Few-shot: `tables/fewshot_mvtec/`
- Strict fixed-threshold metrics: `tables/strict_fixed_threshold/`
- Repair quality and SDR-GT audit:
  `tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json`
- Optional AD2 local archive:
  `submissions/mvtec_ad2_seed7_model256_metadata/submission_protocol.json`
- Seven-baseline comparison: `tables/external_baseline_comparison/`
- Paired inference:
  `tables/external_baseline_comparison/table_paired_inference.csv`
- Worst-category analysis:
  `tables/external_baseline_comparison/table_worst_category_losses.csv`
- Image-score aggregation audit:
  `tables/image_score_aggregation_mvtec15/`
- HN-SEV retention/calibration audit:
  `tables/hn_sev_retention_calibration/`
- HN-SEV input-ablation coverage audit:
  `tables/hn_sev_input_ablation/`
- Repair executor ablation:
  `tables/repair_executor_ablation/`
- Deployment latency smoke audit:
  `tables/deployment_latency/`
- Frozen failure cases:
  `tables/failure_case_panel_mvtec15/`
- RD4AD and SimpleNet imports:
  `tables/official_{rd4ad,simplenet}_mvtec15/`
- Official baseline readiness: `tables/official_baseline_readiness/`
- DiffusionAD measured compute plan: `tables/diffusionad_compute_plan/`
