# Claim Freeze: 2026-06-21

This file turns the follow-up plan into an executable paper boundary for the
current repository.

## Main Direction

Feature-first Lite-SEER-AD:

1. label-free policy and fixed-threshold selection;
2. HN-SEV false-positive suppression;
3. LC-RDS budgeted local repair scheduling.

The main detector is the frozen feature-first heatmap. Regional verification,
repair, and visualization modules are evaluated separately.

## Main Results Allowed

- Main mask metrics must come from `synthetic_normal_fixed_threshold_v1`.
- `oracle_*` metrics are supplement-only diagnostic upper bounds.
- Selection inputs may include normal statistics, deterministic synthetic
  anomalies, normal FPR, synthetic Pixel AP/Dice/AUPRO, latency, and
  augmentation stability.
- Selected candidates must record `uses_real_anomaly_labels=false` and
  `uses_real_anomaly_masks=false`.
- External comparisons must distinguish official/reference artifacts from
  local `*-Lite` controls.

## Claims Not Allowed

- Universal SOTA.
- Diffusion as the main detector.
- CRV as core counterfactual detection evidence.
- CRV score drop as real-defect mask alignment or semantic repair proof.
- Test-GT or oracle threshold results as main deployable metrics.
- Smoke, partial, or `DiffusionAD-Lite` runs as official DiffusionAD.

## Required P0 Additions Still Needing Experiment Output

- HN-SEV TP retention, ROI recall, calibration, and hard-negative taxonomy.
- LC-RDS budget sweep over 10/25/50/75/100/150 ms with violation rate.
- Diffusion necessity comparison against non-diffusion repair executors.
- Public prediction arrays and fixed-threshold JSON bundle for all 33
  categories.

These are experiment artifacts, not narrative changes. If any result is
negative, it should be reported as a limitation rather than hidden.
