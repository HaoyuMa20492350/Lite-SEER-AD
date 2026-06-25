# Dataset Notes

## Supported Benchmarks

- MVTec AD: 15 categories.
- VisA: 12 categories.
- MPDD: 6 categories.
- MVTec AD 2: optional supplementary asset only.

Dataset files are not redistributed in this repository. Follow the licenses
and access rules from the original dataset providers.

## Paper Protocol

For each category, Lite-SEER-AD freezes policy and threshold selection using:

- retained normal images;
- deterministic synthetic anomaly masks;
- normal-pixel false-positive rate;
- synthetic Pixel AP, Dice, and AUPRO;
- measured latency;
- augmentation stability.

The protocol forbids real anomaly labels, real anomaly masks, and held-out test
labels for choosing a category policy or pixel threshold. Held-out real
anomaly masks are used only after freezing for final evaluation, oracle
diagnostics, and post-hoc failure analysis.

## External Baselines

The strict MVTec AD comparison includes seven complete external/reference
baselines. Local `*-Lite` baselines are engineering controls and must not be
reported as official SOTA reproductions.
