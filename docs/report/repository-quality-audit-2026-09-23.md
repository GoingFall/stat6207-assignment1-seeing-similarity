# Repository Quality Guard audit — 2026-09-23

## Conclusion

Current source is not ready for a clean rerun: compilation fails, several Version 1.0 entrypoints have undefined path constants, and migration compatibility/integrity checks have gaps. Retained Version 2.1 experimental artifacts and the frozen release archive still pass their numerical/hash checks. These conclusions concern different guarantees.

This is an audit-only task. No experiment implementation, historical hash, migration annex, tag or release was repaired or rewritten to obtain a passing result.

## Installation and scope

- RQG version: 0.20.4; upstream commit `f01bade7800fc768162938839bd09eebf318f461`.
- Global skill: `~/.agents/skills/repository-quality-guard/` (repository clone).
- Isolated Python runtime: `~/AppData/Local/repository-quality-guard/venv/`; locked dependencies installed successfully. Experimental Conda dependencies were not changed.
- Skill integrity: PASS, 71 protected files. Generic policy; no rules disabled.
- Fixed project baseline: `d6a3968a691555e311dc6186e967648b5720f652`, the committed migration present when this task started. Contrary to the previous conversation state, migration was already committed.
- Commands: `doc-generate`, `doc-search . provenance`, `audit . --diff-base <baseline>`, and final read-only `verify` against the same baseline. Invoke the launcher with Python `-B` on this Windows host; its automatic exec restart otherwise returned before reliable output/exit reporting.
- Generated interface catalog: `docs/api-reference/`. Raw machine report: root `修改说明.md`.
- This global installation is local tooling, not a project-vendored runtime. No pre-push hook was installed: the upstream hook hardcodes a project-local `.agents/skills/repository-quality-guard` runtime that this global installation does not provide. Git delivery enforcement is therefore not installed.

## Confirmed findings

### High — preparation pipeline does not compile

`pipelines/v21_prepare.py:100–105` places the provenance-annex handling outside `main()`, then indents the final print unexpectedly. `python -B -m compileall -q src pipelines checks reporting tools` fails with `IndentationError` at line 105. The workflow `.github/workflows/release-contracts.yml:33` compiles these modules, so the current source cannot pass that CI step. The complete saved-artifact verifier does not import this producer and therefore does not detect the syntax failure.

Repair must preserve the original pre-training `code` snapshot and explicitly document any follow-up source/annex changes; simply changing the certified post-migration hash is not an audit.

### High — Version 1.0 consumers reference an undefined result root

Ruff F821 identifies 23 undefined `R1` references across eight files: `checks/browser_check.py`, `checks/delivery_check.py`, `checks/review_report.py`, `checks/verify.py`, `pipelines/label_audit.py`, `pipelines/supplement.py`, `reporting/explain.py`, and `reporting/export_heatmaps.py`. These modules use `R1` without importing it. Direct execution of `python -m checks.review_report` reproduces `NameError: name 'R1' is not defined` at line 8 before report rendering.

### Medium — legacy path resolver is not idempotent and mappings are incomplete

`src/paths.py:42–60` matches `site/` even when passed an already migrated path. Reproduction: `resolve('site/v1.0/heatmaps/index.json')` returns `site/v1.0/v1.0/heatmaps/index.json`. It also maps `report.pdf.bak` as though it were the exact `report.pdf` filename. Legacy `data/protocol.json` and `data/results/metrics.json` are not mapped, and `backup/pilot` becomes `archive/pilot` rather than the documented dated archive path. Use component-aware, most-specific mappings and recognize current roots before legacy rewriting. The successful V2.1 verifier exercises the V2/V21 mappings, not all of these cases.

### High — historical content verification excludes its own baseline files

`checks/v21_summarize_historical.py:12–14` runs `git diff` against the pre-migration commit but restricts the pathspec to new directories. In that baseline those directories do not exist; the current artifacts appear as additions. The next line discards every `A` and every `R`, including a potentially content-changing rename. The exact command returns additions for all current V2 data/results/config artifacts. Thus this gate does not establish historical byte identity. Compare explicit old-path Git blobs to mapped current files, or use the original content manifests; do not infer identity from ignored diff statuses. This finding does not assert the current data have actually been corrupted.

### Medium — new-layout packaging and archive verification disagree

`tools/package_v21.py:40–41` collects current `configs/v2.1`, `data/v2.1` and `results/v2.1` entries. Its final verifier, `checks/v21_release.py:24–25`, unconditionally reads the old `data_v21` and `configs/v21` entry names. A newly built current-layout archive would fail that verifier. Existing immutable archives correctly pass. No fresh package was created, and frozen archives were not removed to exercise the new-build path. Use explicit archive-layout detection or a separately versioned build/verification contract.

## Verification performed

- RQG dependency installation and protected runtime integrity: passed.
- RQG scan: 1,506 findings, comprising 115 Critical, 889 Error, 95 Warning and 407 Info. Relative to the fixed baseline, ordinary C/E/W deltas are zero; these are rule hits, not counts of confirmed bugs.
- RQG audit: `UNVERIFIED`, static status `REJECT`, 51 semantic delta candidates and 236 report-completeness issues. Many candidates concern legitimate experiment/release identities; no blanket acceptance or suppression was applied.
- Python compilation: failed on `pipelines/v21_prepare.py`.
- Ruff F821: 23 undefined-name hits plus two syntax diagnostics in the same preparation file.
- `checks.review_report`: failed with the reproduced `R1` NameError.
- `checks.v21_contracts`: passed.
- `checks.v2_ci`: passed, 14 contracts.
- `checks.v21_verify`: passed; 60,295 unique image contents, 16 models, 15 long-tail runs and retained-artifact/annex checks.
- `checks.v21_release`: passed, 109 archive entries; SHA-256 `425ae9c9f3871e372d0f2cf79f41d12ef5b2eabc6463e4fbb32dcef6c4ad77db`.

## Report-gate interpretation

The root report is the unmodified machine-generated audit ledger with pending semantic/architecture/reduction/commit fields. It is not a completed acceptance certificate. The independent source review above establishes concrete failures, so there is no basis for declaring the repository accepted. All 51 semantic candidates have not been individually adjudicated, and no claim of exhaustive manual review is made. The final `verify` output must be read together with this limitation: report incompleteness is separate from the reproduced code failures.

Priority for follow-up: restore compilability and missing path imports; repair resolver behavior and historical identity comparison; align packaging with its archive verifier. Then record the reviewed migration correction without replacing pre-training evidence and rerun compilation, affected legacy entrypoints, full artifact verification and the RQG report workflow.
