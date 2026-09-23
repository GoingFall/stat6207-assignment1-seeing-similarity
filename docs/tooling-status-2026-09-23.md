# Tooling status — 2026-09-23

| Item | Status | Evidence / boundary |
|---|---|---|
| Full V2.1 replay preflight | `ready` | `checks.local_replay_preflight`: source image count missing `0`, embeddings directory present, config hash matches |
| V2.1 retained-artifact verification | passed | `python -m checks.v21_verify` |
| Version 2.0 CPU contracts | passed | `python -m checks.v2_ci`, 14 contracts |
| V2.1 regression contracts | passed | `python -m checks.v21_contracts` |
| V2.1 public archive audit | passed | `python -m checks.v21_release`, 109 entries |
| Ruff availability | installed | `ruff==0.15.20` in `requirements.txt` and `stat6207-a1` |
| Ruff full-repository style check | existing debt remains | 631 findings; no bulk autofix was applied |
| Python compilation | passed | `python -B -m compileall -q src pipelines checks reporting tools` |
| Version 1.0 report review entrypoint | passed | `python -m checks.review_report` |
| RQG global skill | installed | `C:\Users\10489\.agents\skills\repository-quality-guard` |
| RQG project-local deployment | blocked by upstream integrity | `v0.20.4` checked-out `scripts/quality_guard.py` does not match its own `runtime/MANIFEST.sha256`; deployment correctly refuses to bypass the seal |
| pre-push hook | installed locally | `.git/hooks/pre-push` invokes the global RQG runtime; Git hooks are not versioned |

The pre-push hook is intentionally documented as machine-local until a verified RQG distribution can be deployed into `.agents/skills/repository-quality-guard`. No release hash, pre-training provenance, or experiment result was rewritten to obtain these statuses.
