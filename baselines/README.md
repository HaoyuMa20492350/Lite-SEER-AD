# Baseline Integration Contract

The main pipeline is implemented first. Baselines should be integrated as adapters that export the same inference artifacts:

- `predictions.npz` with `labels`, `image_scores`, `masks`, `heatmaps`
- `scores.csv`
- `metrics.json`

Planned adapters:

- PatchCore
- PaDiM
- SimpleNet
- DRAEM
- RD4AD
- UniAD
- DiffusionAD
- DDAD
- InvAD, optional strengthening baseline

The runners in `baselines/run_baseline.py` are local engineering controls, not
official reproductions. Tables must display them as `PatchCore-Local`,
`PaDiM-Local`, `SimpleNet-Lite`, `DRAEM-Lite`, `RD4AD-Lite`, `UniAD-Lite`,
`DiffusionAD-Lite`, and `DDAD-Lite`. Their results cannot be used to claim
official method performance or external SOTA superiority.

Third-party methods can be exported to `predictions.npz` with the arrays listed
above, then imported with `tools/import_external_baseline.py` or placed under
`baselines/external_outputs/<method>/<category>/predictions.npz` for
`tools/run_mvtec15_baselines.py`.

Every row carries `implementation_variant`, `official_implementation`,
`source_path`/`source_url`, `source_commit`, and `reference_key`. An imported
artifact is still unverified unless `--official-implementation` and exact
source provenance are supplied.

Pinned author repositories and commits are recorded in
`baselines/official_sources.json`. External artifacts use:

`baselines/external_outputs/<dataset>/<method>/<category>/predictions.npz`

with a sibling `provenance.json`. Run
`python tools/audit_official_baseline_readiness.py` before importing. The
importer rejects an official claim if repository, commit, source kind, dataset,
or category differs from the pinned manifest.

Use `tools/audit_mvtec15_baseline_coverage.py` after exporting tables. The
publishable MVTec15 comparison is incomplete until
`baseline_coverage.json` reports `complete: true` for
`patchcore,padim,simplenet,draem,rd4ad,uniad,diffusionad,ddad`.

The pinned official PatchCore path is complete:

```powershell
python tools/fetch_official_baseline_sources.py
python tools/audit_official_source_environments.py
python tools/materialize_patchcore_pretrained.py
python tools/run_official_patchcore.py
python tools/run_mvtec15_baselines.py `
  --methods patchcore `
  --prefer-external `
  --out tables/official_patchcore_mvtec15
python tools/export_official_patchcore_comparison.py
```

The resulting 15/15 comparison is under
`tables/official_patchcore_comparison/`. It is mixed rather than an overall
win: Lite-SEER-AD has higher mean AUPRO, while official PatchCore has higher
mean Image AUROC, Pixel AUROC, Pixel AP, and fixed Dice.

The other executable pinned adapters use:

```powershell
python tools/run_reference_padim.py --categories all --resume
python tools/run_official_uniad.py --categories all --resume
python tools/run_official_draem.py --categories all --resume
python tools/materialize_ddad_pretrained.py
python tools/run_official_ddad.py --categories all --resume
python tools/run_official_rd4ad.py --categories all --resume
python tools/run_official_simplenet.py --categories all --resume
```

RD4AD and SimpleNet do not have author-released MVTec checkpoints. Their
adapters train the pinned author architectures to a fixed final epoch and do
not evaluate the test set during training. This intentionally removes the
test-set model selection present in the author scripts. Both adapters export
raw maps, resumable training checkpoints, and the same synthetic-normal fixed
threshold evidence as the pretrained adapters.

DiffusionAD also lacks released checkpoints. Prepare its author-released
foreground masks with:

```powershell
python tools/materialize_diffusionad_foregrounds.py

# One-batch plumbing and memory check; never paper eligible.
python tools/run_official_diffusionad.py `
  --categories bottle --epochs 1 --batch-size 1 `
  --max-train-batches 1 `
  --max-normal-images 1 --synthetic-variants 1 --synthetic-seeds 7 `
  --external-root baselines/external_outputs_smoke_diffusionad

# Full fixed-final-epoch reproduction. This is a multi-day job.
python tools/run_official_diffusionad.py --categories all --resume
```

The materializer validates all per-image masks against the MVTec training
split. The official configuration requires 3000 epochs per category and about
678,000 optimization steps for MVTec15, so a complete from-scratch result is
not represented by the lightweight local DiffusionAD control. See
`docs/diffusionad_reproduction_status_zh.md`.

Current MVTec15 readiness is `105/120`: PatchCore, PaDiM, UniAD, DRAEM, DDAD,
RD4AD, and SimpleNet are complete for all 15 categories. The 15 missing rows
are exactly the full DiffusionAD runs. Its one-batch real smoke test passes,
but is explicitly marked `paper_eligible_full_training=false`.

For publishable tables, use official external implementations under the same
image size, category split, fixed-threshold protocol, and metric implementation
used by `evaluate.py`.

New local runs use `synchronized_batch_latency_v1`: batch size 1 by default,
50 warmups, 200 measurements, and CUDA synchronization around each measured
call. Historical rows without `latency_protocol` are legacy measurements and
must not be used for cross-method efficiency claims.
