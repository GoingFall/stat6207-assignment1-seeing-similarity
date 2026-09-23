# Post-migration verification — 2026-09-23

Verification ran on the original Windows host in the `stat6207-a1` Conda environment, which includes Faiss and PyTorch.

## Corrections found during execution

The full verifier initially failed on two dynamically loaded historical paths. In `checks/v21_verify.py`, JSON loading and selected-checkpoint loading now use `src.paths.resolve()` rather than concatenating the repository root with the recorded legacy path. These are verifier path corrections, not experiment changes.

No pre-training source hash or migration-annex hash was refreshed to accommodate these fixes. The verifier is not one of the 35 files covered by the existing migration annex. The `code` object in `results/v2.1/provenance.json` was independently compared with the committed original at `HEAD:results_v21/provenance.json` and is identical.

## Executed checks

- `python -m checks.v21_verify`: passed; regenerated `results/v2.1/verification.json`. Covers 60,295 unique image contents, 16 selected models, 15 long-tail runs, five seeds, the migration annex, calibration/OOD/monitoring recomputation and historical release hashes.
- `python -m checks.v21_contracts`: passed.
- `python -m checks.v2_ci`: passed, 14 contracts.

These commands verify retained artifacts; they do not retrain the experiments or rebuild frozen releases. The
migration and these follow-up corrections are committed together on `main`, so the layout described in
`docs/MIGRATION-2026-09-23.md` is the layout this verification passed against. `releases/1.0/`,
`releases/2.0/` and `releases/2.1/` were neither rebuilt nor renamed by any of it.

## Global naming guidance

The user-level skill `repository-naming` is installed at `~/.agents/skills/repository-naming/SKILL.md` and was discovered by OpenCode. It covers purpose/version directory layout, language-appropriate filenames, dated archives, centralized path resolution, evidence-preserving migrations and end-to-end verification. Skill discovery is relevance-based guidance, not an always-on enforcement hook.
