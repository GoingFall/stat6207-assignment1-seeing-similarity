# Version 2.1 revision release

This revision supersedes the strict independence and group-F1 claims of 2.0 for iNaturalist. Version 2.0 remains historical; no historical ZIP, data lock, result or checksum is replaced.

## Corrections

- Content-based duplicate removal and full quarantine of conflicting labels; original split memberships retained, known classes subsampled to 20/8/8 by image ID.
- Group macro-F1 counts false positives from every true-class group. Group recall is class-macro averaged.
- Balanced probe plus five-seed cross-entropy, weighted and balanced-sampler training rerun with validation-selected learning rate/weight decay. Every selected checkpoint and train/validation/test prediction artifact is retained locally and hashed.
- Calibration, near-OOD and fixed-reference/matched-window monitoring rerun on revised splits.
- Deterministic exact-search boundary ties corrected and regression-tested. Historical ANN latency/recall are not presented as new 2.1 measurements.
- Public release uses an explicit allowlist and excludes session transcripts and nested releases.

## Evidence and limitations

Read `docs/report/V21-RESULTS.md` and `results_v21/verification.json`. Exact 50:1 remains infeasible; far-OOD remains blocked. No WSL GPU benchmark is claimed.

This is an audit-driven revision using previously evaluated data. It is not a new untouched holdout. Byte-content de-duplication does not exclude visually near-identical images or pretraining overlap. The original encoder revision is unknown; reproducibility is conditional on verified original feature artifacts. Source feature hashes and stage-specific source/config/environment provenance are retained.

## Publication boundary

Only code, configurations, redacted image-ID/content-hash manifests, aggregate results, permitted figures and provenance are packaged. Raw images, local image filenames, embeddings, selected checkpoints, per-example predictions, conversation transcripts and nested ZIPs are excluded. Private artifacts are verified locally, not redistributed.

The adjacent `MANIFEST-2.1.json` describes all payload files. `Assignment1-2.1.zip.sha256` contains the complete archive SHA-256. Run `python -m checks.v21_release` for a data-free package audit.
