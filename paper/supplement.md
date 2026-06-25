# Lite-SEER-AD Supplementary Material Draft

This supplement is a claim-bounded companion to the manuscript. It is organized
around reproducible evidence rather than additional claims.

## S1. Label-Free Selection Protocol

- Fixed policy and threshold bundle:
  `artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json`
- Main 33-category evidence package:
  `tables/feature_first_fusion_aggregate_paper_package/`
- Image-score aggregation negative audit:
  `tables/image_score_aggregation_mvtec15/`

Selection uses retained normal images and deterministic synthetic defects. Real
anomaly labels and masks are reserved for held-out audit only.

## S2. External Baseline And Statistical Audit

- Seven-baseline comparison:
  `tables/external_baseline_comparison/`
- Paired BCa bootstrap, exact sign tests, and Holm correction:
  `artifacts/stats/`
- DiffusionAD compute plan:
  `tables/diffusionad_compute_plan/`

DiffusionAD full official training is a conditional P2 item. Smoke or partial
training runs are not paper-eligible external-baseline evidence.

## S3. Regional Module Audits

- HN-SEV retention, calibration, and taxonomy audit:
  `tables/hn_sev_retention_calibration/`
- HN-SEV input-ablation coverage audit:
  `tables/hn_sev_input_ablation/`
- LC-RDS budget replay audit:
  `tables/lc_rds_budget_audit/`
- Repair executor ablation:
  `tables/repair_executor_ablation/`

HN-SEV is currently limited to false-positive suppression because TP retention
is not recall-safe. LC-RDS budget evidence now includes five-action
config/code coverage, offline replay, measured synthetic action execution,
measured-latency ROI replay, and a guarded 198-run production six-budget sweep.
Diffusion repair is treated as a replaceable executor because the same-protocol
clean-target Pareto audit finds `diffusion_not_necessary` against the strongest
non-diffusion baseline.

## S4. Deployment And Reproducibility

- Deployment latency smoke audit:
  `tables/deployment_latency/`
- Production inference component latency audit:
  `tables/deployment_production_latency/`
- Public release readiness:
  `tables/release_readiness/`
- SHA256 artifact manifest:
  `artifacts/manifest.json`
- Prediction array manifest:
  `artifacts/predictions_manifest.json`

The deployment evidence now includes both synchronized component smoke timing
and 33-category production inference component timing. It also includes the
full `10/25/50/75/100/150 ms` production LC-RDS budget matrix:
`budget_runs=198/198`, `missing_budget_runs=0`, all five required actions
observed, and maximum budget violation rate below 1%. It is still not a final
cross-hardware deployment claim because only the current hardware profile has
production timing and energy evidence.
Public release is gated by final GitHub Release, Zenodo DOI, and Hugging Face
URLs.

## S5. Claim Boundaries And Submission Materials

- Frozen claim boundary:
  `docs/claim_freeze_2026_06_21.md`
- Results and limitations draft:
  `docs/results_limitations_draft.md`
- Reviewer FAQ:
  `docs/reviewer_faq.md`
- Submission statements and cover letter skeleton:
  `docs/submission_statement_placeholders.md`
- Completion gap matrix:
  `tables/completion_gap_matrix/`

Before upload, convert this supplement to the selected journal template and
replace all author, affiliation, funding, conflict, data/code availability, and
public-release placeholders with final values.
