# Lite-SEER-AD Submission Statements

This file was rendered from `submission_metadata.json` by `tools/render_submission_statements.py`.

## Target Journal

- Journal: Pattern Recognition
- Publisher/platform: Elsevier / ScienceDirect
- Article type: Full Length Article / Original Research
- Template status: Elsevier article template selected; manuscript source remains in Markdown until author metadata and final figures are frozen
- Required word/page limit: submit as single-column, double-spaced manuscript with numbered pages; Pattern Recognition guide indicates no less than 20 and no more than 35 pages for regular submissions
- Supplement format: separate supplementary material package with reproducibility tables, FAQ, limitations, and artifact manifest
- Guide source checked: https://www.journals.elsevier.com/pattern-recognition

## Authors And Affiliations

- Corresponding author: <name>, <email>, <affiliation>
- Author list: <ordered author names>
- Affiliations: <institution, department, city, country>
- ORCID IDs: <optional or required ORCID IDs>

## Funding Statement

<exact funding agency names, grant numbers, and recipient authors, or the journal-required no-funding wording>

## Conflict Of Interest Statement

<exact conflict-of-interest statement, or the journal-required no-conflict wording>

## Data Availability Statement

GitHub Release URL: <github release url>. Zenodo DOI: <zenodo doi>. Hugging Face URL: <hugging face model or dataset url>. The benchmark datasets used in this study are publicly available from their original providers subject to their licenses. Dataset download instructions are provided in DATASETS.md.

## Code Availability Statement

repository URL: https://github.com/HaoyuMa20492350/Lite-SEER-AD. The source code, experiment configurations, table regeneration scripts, environment files, Dockerfile, CI workflow, and SHA256 artifact manifest are available under <license>. The exact release commit is <commit sha>.

## Reproducibility Statement

All retained quantitative claims are tied to fixed configs, seeds, threshold policies, prediction arrays, and table-generation scripts. The selection policy uses normal images and synthetic anomalies only; real anomaly labels and masks are reserved for held-out audit. The release manifest reports SHA256 hashes for the reproducibility artifacts.

## Ethics Statement

This study evaluates industrial anomaly detection datasets and does not involve human participants, human tissue, personal data, or animal experiments.

## Author Contributions

- Conceptualization: <names>
- Methodology: <names>
- Software: <names>
- Validation: <names>
- Formal analysis: <names>
- Investigation: <names>
- Data curation: <names>
- Writing - original draft: <names>
- Writing - review and editing: <names>
- Visualization: <names>
- Supervision: <names>
- Funding acquisition: <names>

## Cover Letter Skeleton

Dear Editor,

We submit `Lite-SEER-AD: Label-Free Feature-First Industrial Anomaly Localization with Selective Regional Verification` for consideration as <article type> in <journal>.

The manuscript presents a feature-first industrial anomaly localization pipeline centered on label-free policy/threshold selection, HN-SEV false-positive suppression, and LC-RDS budgeted local repair scheduling. We explicitly do not claim universal SOTA, do not use real anomaly labels or masks for policy selection, and downgrade CRV to repair visualization/post-hoc audit based on negative alignment evidence.

All retained claims are backed by reproducible configs, fixed thresholds, prediction manifests, statistical tables, release metadata, and the final public artifact links.

Sincerely,

<corresponding author name>

## Final Pre-Upload Checks

- Confirm title, abstract, contribution list, figures, supplement, and cover letter all follow the feature-first mainline.
- Verify CRV is never described as a main contribution.
- Verify DiffusionAD full is either absent from claims or clearly marked as optional/pending official reproduction.
- Re-run `python tools/export_submission_package_readiness.py`, `python tools/export_completion_gap_matrix.py`, and `pytest -q`.
