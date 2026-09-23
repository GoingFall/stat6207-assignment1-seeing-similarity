# Repository layout normalisation, 2026-09-23

This change gives every versioned area one naming scheme, `vMAJOR.MINOR`. No experiment, metric, figure,
lock or frozen release archive was modified; this is a pure path and packaging-metadata change.

## Why

Three naming conventions coexisted: `data_v2/` (underscore), `configs/v2/` (slash) and `releases/2.0/`
(dotted). `v21` was ambiguous, appearing as `data_v21/` in directories, `2.1` in `VERSION` and
`releases/2.1/` in releases. Version 1.0 had no version marker at all: `data/`, `site/` and `report.pdf`
sat in the project root and the README explained in prose that they were "the historical Version 1.0
assignment", so the layout could not be read from the names. `backup/` mixed event-named snapshots
(`website-before-interaction-update`) with version-named ones (`v3-before-reorganization`), and the `v3`
label contradicted `VERSION`, since that historical `v3` draft is what shipped as Version 1.0.

## Rename map

| Before | After |
|---|---|
| `data/` | `data/v1.0/` |
| `data/results/` | `results/v1.0/` |
| `data_v2/` | `data/v2.0/` |
| `results_v2/` | `results/v2.0/` |
| `data_v21/` | `data/v2.1/` |
| `results_v21/` | `results/v2.1/` |
| `configs/v2/` | `configs/v2.0/` |
| `configs/v21/` | `configs/v2.1/` |
| `report.pdf` (root) | `report/v1.0/report.pdf` |
| `site/` | `site/v1.0/` |
| `backup/` | `archive/` |
| `backup/pilot/` | `archive/2026-09-21-pilot-270-image/` |
| `backup/v3-before-reorganization/` | `archive/2026-09-21-v1.0-before-source-reorganization/` |
| `backup/before-normalization-clarification/` | `archive/2026-09-21-v1.0-before-normalization-clarification/` |
| `backup/website-before-interaction-update/` | `archive/2026-09-21-v1.0-website-before-interaction-update/` |
| `.github/workflows/v2-cpu-contracts.yml` | `.github/workflows/release-contracts.yml` |

Frozen releases are deliberately excluded from the map. `releases/1.0/`, `releases/2.0/` and `releases/2.1/`
keep their original directory and file names, because `results/v2.1/provenance.json` records those paths
together with their hashes.

## Code

`src/paths.py` is now the single source of truth: `V1`/`V2`/`V21`, `R1`/`R2`/`R21`, `DATA`, `RESULTS`,
`CONFIG`, `SITE`, `REPORT`, `ARCHIVE`, `RELEASES`, derived from `VERSION`. `OUT` and `OUT_V2`/`OUT_V21` are
retained as aliases so release-scoped scripts keep their original variable names.

`src.paths.resolve()` maps the pre-migration names to their new locations. Frozen artifacts recorded paths
such as `data_v2/manifests/sift1m_lock.json` or `site/heatmaps/index.json` inside their JSON, and rewriting
those certified files would have destroyed their hashes, so the resolution happens at read time instead.

## Audit evidence

`results/v2.1/provenance.json` records the certified pre-training source snapshot under `code`. Those hashes
are unchanged. A new `layout_migration` annex records, for every file this change touched, both the
pre-migration hash taken from the certified snapshot and the post-migration hash:

```json
"layout_migration": {
  "scope": "...",
  "baseline_commit": "2fa9424a2b1ccda356028f2d3caa6d99d5814a89",
  "training_critical_changed": ["..."],
  "training_critical_unchanged": ["..."],
  "files": {"pipelines/analyze.py": {"pre_migration_sha256": "...", "post_migration_sha256": "..."}}
}
```

`checks.v21_verify` validates every annex hash and then accepts, for each training-critical file, only the
certified pre-training hash or the annexed post-migration hash. Any further edit to a training-critical file
still fails verification. `pipelines/v21_prepare.py` carries the annex forward if it is regenerated, so a
stale annex fails loudly rather than being silently dropped.

`training.json`, `training_partial.json` and `open_set.json` were re-linked to the amended provenance
document, because their `provenance_sha256` / `training_result_sha256` fields hash that document's bytes.

To audit the change by hand:

```powershell
git diff 2fa9424a2b1ccda356028f2d3caa6d99d5814a89 -- pipelines/ checks/ tools/ src/
```

## Version control policy

- `VERSION` states the current release. Packagers compare versions numerically, so building the Version 2.0
  release no longer requires `VERSION` to read exactly `2.0`. Each release directory carries its own
  `VERSION`.
- Frozen archives under `releases/<version>/` are never overwritten: each packager raises if its archive
  already exists.
- `data/v1.0/provenance/` is tracked, because `pipelines.prepare_data` and `pipelines.supplement` read the
  pilot manifest and the 20 Animals-10 probe images from it.
- The JSON records under `results/v1.0/` are tracked as Version 1.0 evidence.
- No path that was tracked before this change is ignored after it. The rewritten `.gitignore` was checked
  path by path against `git ls-tree -r <previous-commit>`; the first draft wrongly ignored
  `data/v2.1/manifests/lock.json` and the five `data/v2.1/manifests/long_tail/*.json` manifests, which were
  tracked before and are tracked again.
- Large inputs, embeddings, generated media and the 740 MB Version 1.0 submission archive stay out of git;
  their SHA-256 sidecars and manifests are tracked so integrity remains verifiable.
- Historical documents (`docs/session/`, `docs/report/V2-EXECUTION-LOG.md`, `docs/report/V2-RESULTS.md`,
  `docs/report/V21-RESULTS.md`, `docs/report/REPORT-REVISION.md`, `docs/releases/RELEASE-1.0.md`) keep the
  paths that were current when they were written. They are records, not instructions.
