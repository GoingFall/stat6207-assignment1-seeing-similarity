# Documentation Index

This directory contains active project documentation. The root `README.md` is the short entry point; use this index for the detailed records.

## Categories

- `course/`: assignment instructions and supplied lecture material. These are source materials, not generated project claims.
- `protocol/`: the executed Version 1.0 experiment protocol and its scientific limitations.
- `plans/`: reviewed plans and implementation protocols. Version 2.0 execution is active; completed gates and corrections are recorded in `report/V2-EXECUTION-LOG.md`.
- `report/`: generated report text and report-revision records. The rendered PDF lives at `report/v1.0/report.pdf` and in `site/v1.0/report.pdf`.
- `releases/`: submission pointers and release notes. Immutable release artifacts remain under `releases/<version>/`.
- `session/`: exported OpenCode collaboration transcript in readable Markdown and full JSON forms. Paths and commands inside the transcript are historical snapshots and are intentionally not rewritten after files move.

## Which document to use

- Reproduce the locked result: start with `../README.md`, then `protocol/Revised Experiment Protocol.md`.
- Review the submitted release: use `releases/SUBMISSION.md` and `releases/RELEASE-1.0.md`.
- Inspect report wording or revision history: use `report/report_text.md` and `report/REPORT-REVISION.md`.
- Review the Version 2.0 scope and current execution: use `plans/PLAN-2.0-ANN-OPENSET.md` and `report/V2-EXECUTION-LOG.md`.
- Trace collaboration decisions: use `session/session-transcript.md`.
- Audit the repository layout change: use `MIGRATION-2026-09-23.md`.

Historical documents remain inside `archive/` and are not active project documentation. `MIGRATION-2026-09-23.md` records the path-rename map, the tracked/ignored policy and the audit evidence.

The immutable Version 1.0 archive and its authoritative manifest remain under `../releases/1.0/`; documents in this directory describe the active workspace without replacing that release record.
