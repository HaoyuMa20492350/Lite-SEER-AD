# Lite-SEER-AD Model Card

## Summary

Lite-SEER-AD is a feature-first industrial anomaly localization pipeline. The
paper-facing detector uses frozen normal-feature priors and label-free
policy/threshold selection from normal images and deterministic synthetic
defects. HN-SEV, LC-RDS, and CRV are evaluated as regional modules and do not
change the frozen main detector scores.

## Intended Use

- Industrial anomaly localization research.
- Label-free pixel-policy and threshold selection studies.
- False-positive filtering and budgeted local repair analysis.

## Out Of Scope

- Safety-critical deployment without independent validation.
- Claims of universal SOTA across industrial anomaly benchmarks.
- Use of CRV score drops as proof of semantic defect removal or GT-aligned
  repair.
- Treating smoke or partial DiffusionAD runs as official DiffusionAD results.

## Training And Selection Data

Policy and threshold selection may use retained normal images, deterministic
synthetic anomaly masks, normal false-positive rate, synthetic Pixel AP/Dice/
AUPRO, latency, and augmentation stability. It must not use real anomaly
labels, real anomaly masks, or held-out test labels for category decisions.

## Evaluation

The paper package covers MVTec AD, VisA, and MPDD over 33 categories and three
held-out seeds. Main deployable mask metrics use
`synthetic_normal_fixed_threshold_v1`; `oracle_*` metrics are diagnostic upper
bounds only.

## Limitations

The current evidence supports label-free selection, strict fixed-threshold
behavior, HN-SEV false-positive suppression, and LC-RDS budget analysis. It
does not support universal SOTA, image-level improvement claims, or positive
CRV-GT alignment.
