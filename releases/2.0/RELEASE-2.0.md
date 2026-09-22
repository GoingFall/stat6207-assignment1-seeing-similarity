# Release 2.0

Version 2.0 adds independently locked SOP semantic retrieval, native-Windows SIFT1M ANN systems benchmarking, controlled iNaturalist Birds long-tail training, calibrated near-OOD evaluation, drift monitoring, CPU CI contracts and a health/status service.

See `docs/report/V2-RESULTS.md` for the result summary and `docs/report/V2-EXECUTION-LOG.md` for protocol corrections and full execution provenance.

## Publication boundary

The release archive contains code, configurations, documentation, hashes/manifests, aggregate JSON metrics and figures. It excludes source images, embeddings, model checkpoints, Faiss indexes and per-query neighbour/score matrices. The public iNaturalist image manifest removes local paths and source filenames.

## Known limitations

- The exact 50:1 long-tail experiment is infeasible with the leakage-free 30-image training pool and remains explicitly blocked.
- Near-OOD class-novel bird rejection is weak; no strong open-set detection claim is made.
- No verified far-OOD dataset was used.
- The optional WSL2 GPU Faiss smoke test was not run and is not required for the native Windows CPU result.
- Hardware timings are machine-specific and large-data/GPU runs are not executed in CI.
