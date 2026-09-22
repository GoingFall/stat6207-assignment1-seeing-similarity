# Assignment 1 — version 1.0

Archived release: 21 September 2026.

## Deliverables

- `Assignment1-submission.zip`: complete version 1.0 submission snapshot.
- `Assignment1-submission.zip.sha256`: archive checksum.
- `MANIFEST-1.0.json`: size and SHA-256 for every payload file; also included inside the archive. The manifest excludes itself and the outer ZIP/checksum to avoid circular hashes.
- `report.pdf`: 11-page rubric-aligned report; identical copy in `site/report.pdf`.
- `site/`: static explorer, pairwise ResNet Grad-CAM, UMAP explanations and `ood.html`.
- Source scripts, locked data/results, environment specifications and session transcript support reproduction.

## Included revisions

Explicit raw/unit distance definitions and 21-configuration normalization verification; cat and dog retrieval examples; post-hoc clean RGB-pixel KNN baseline; source-class failure rates; explicit OOD-page pointer. Primary protocol and test locks retain their original hashes. Version 1.0 is the delivery version; protocol v3 is the experiment version.

## Checks and cleanup

Numerical, browser, heatmap, supplement and delivery checks passed. Report rendering and review records are retained in `data/results/report_review/`. Packaging verifies ZIP integrity and every payload hash. Runtime bytecode and transient browser logs are removed; the superseded before-update UMAP screenshot belongs in `backup/`. Historical versions remain under `backup/` and are excluded from the submission.

The archive is the frozen 1.0 snapshot. Root files are the corresponding working copy; future revisions should use a new release version rather than replacing the saved 1.0 snapshot. Public Hugging Face deployment remains pending account authentication.
