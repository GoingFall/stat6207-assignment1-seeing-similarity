# Version 2.1 — content-cleaned, prediction-audited revision

## Scope and protocol

This is an audit-driven revision, not a fresh unseen benchmark. V2.0 remains historical and its release ZIP is unchanged. SOP/SIFT results are inherited historical evidence and have not been rebenchmarked in 2.1. No 2.1 ANN throughput claim is made.

Original class and split memberships are retained. Conflicting-label content is quarantined in full; same-label duplicate bytes are deduplicated using fixed train/validation/test/unknown-development/unknown-test priority and image ID. Known classes use the lowest remaining IDs per split: 20 train, 8 validation, 8 test. Selection never uses predictions. This reduced common budget avoids unequal per-class counts and preserves exact 10:1 sampling (20:2). It is not directly comparable to the original 30/10/10 experiment.

The selected corpus has 60,295 byte-content-unique images; 10 conflicting/duplicate records were removed before budget selection. There are 1,000 known and 243/243 unknown-development/test classes. SHA-256 checks exclude byte-identical duplication only; visually near-duplicate images and pretraining overlap remain unmeasured.

## Training and retained evidence

Linear probes select learning rate (1e-4/3e-4/1e-3), weight decay (1e-4/1e-2) and stopping epoch using validation macro-F1. All 16 selected checkpoints and their train/validation/test logits, labels and predictions are saved locally with hashes. Group macro-F1 uses full-test false positives; group recall is a class-macro average.

- KNN test accuracy: 0.500625; macro-F1: 0.497612.
- Balanced probe test accuracy: 0.481500; macro-F1: 0.461903.
- Balanced probe train/validation/test accuracy: 0.657300 / 0.478625 / 0.481500; train−test gap: 0.175800.

### Controlled long tail: five seeds, mean ± sample standard deviation

- cross_entropy: accuracy 0.221075 ± 0.004461; macro_f1 0.176296 ± 0.005065; tail_macro_f1 0.010979 ± 0.002020; tail_recall 0.006757 ± 0.001273.
- class_weighted: accuracy 0.313675 ± 0.002152; macro_f1 0.277533 ± 0.004261; tail_macro_f1 0.143642 ± 0.007169; tail_recall 0.114039 ± 0.005836.
- balanced_sampler: accuracy 0.346925 ± 0.012737; macro_f1 0.330811 ± 0.013830; tail_macro_f1 0.266462 ± 0.012273; tail_recall 0.300751 ± 0.011984.

Exact 50:1 remains blocked: 20 unique training candidates cannot realize positive integer 50:1 sampling. No replacement or evaluation data borrowing is used.

## Calibration and near-OOD

Temperature fitted on validation only: 0.295945. Test ECE 0.447278 → 0.013775.

- nearest_neighbor: AUROC 0.577794; AUPR-OOD 0.656291; FPR95-OOD 0.916250; known coverage 0.947000; unknown rejection 0.076803.
- neighbour_agreement_margin: AUROC 0.583633; AUPR-OOD 0.647289; FPR95-OOD 0.871625; known coverage 1.000000; unknown rejection 0.000000.
- temperature_max_probability: AUROC 0.599080; AUPR-OOD 0.648362; FPR95-OOD 0.849750; known coverage 0.954625; unknown rejection 0.040089.
- negative_energy: AUROC 0.521674; AUPR-OOD 0.598001; FPR95-OOD 0.911375; known coverage 0.989750; unknown rejection 0.007326.

Highest observed held-out AUROC: temperature_max_probability. This is descriptive comparison, not test-driven model selection. No strong open-set detector or far-OOD claim is made.

## Monitoring

A fixed 2,000-image validation reference is compared with 1,000-image windows throughout. Thresholds use 200 repeated windows from the disjoint remaining validation pool; all comparisons use fixed reference-derived bandwidth. Repeated calibration windows overlap, so no independent 1% false-alarm guarantee is claimed.

- test: 8 nonoverlapping windows; alert rates {"centroid_cosine_shift": 0.0, "projection_psi": 0.0, "rbf_mmd": 0.0}; 0 remainder images not evaluated.
- unknown_test: 12 nonoverlapping windows; alert rates {"centroid_cosine_shift": 1.0, "projection_psi": 1.0, "rbf_mmd": 1.0}; 148 remainder images not evaluated.

## Provenance and reproduction boundary

2.1 subsets individually hash-verified V2 frozen features by image ID; it does not re-encode images. The original encoder weight/processor revision was not recorded in 2.0 and cannot be retroactively certified. Reproducibility is therefore conditional on the hash-locked V2 feature artifacts, not a promise of raw-to-feature bitwise regeneration. Config/split locks were written before retraining. Training-critical source hashes and the Python package inventory are saved in results_v21/provenance.json.

Verification independently checks saved models against test logits, recomputes all 16 models’ classification/group metrics, checks five-seed aggregation, refits temperature, recomputes rejection thresholds/metrics, and recomputes monitoring from embeddings. The exact-search tie regression now returns the same globally lowest tied IDs for multiple shard sizes. Before full verification on a fresh workspace, run `python -m checks.v21_exact_audit` for the SIFT correctness gate (requires original SIFT files).

The release allowlist excludes images, source paths, feature arrays, checkpoints, per-example predictions, conversation transcripts and nested archives. These private artifacts stay local; public manifests retain image IDs/content hashes and permitted labels. A data-free package verifier checks every archive entry and public lock.

## Commands

```powershell
python -m pipelines.v21_prepare
python -m pipelines.v21_train
python -m pipelines.v21_open_set
python -m pipelines.v21_monitor
python -m checks.v21_contracts
python -m checks.v21_verify
python -m reporting.v21_results
python -m tools.package_v21
python -m checks.v21_release
```

Run from the repository root with the original local dataset/feature environment. Existing completed stage artifacts refuse overwrite. Restore historical 2.0 code from its ZIP to rerun the original implementation; current shared metric code includes the 2.1 corrections.
