# Frozen Version 1.0 submission

The submission for this assignment is the immutable archive `releases/1.0/Assignment1-submission.zip`. It is the only release that satisfies every instruction item, because it is the only one that carries the report, the code, the website and the session transcript together.

## What is inside the archive

The archive is flat: paths are relative to the ZIP root, not to this working tree.

| Inside the ZIP | In the active working tree |
| --- | --- |
| `report.pdf` | `report/v1.0/report.pdf` (also `site/v1.0/report.pdf`) |
| `site/index.html`, `site/...` | `site/v1.0/` |
| `data/...` | `data/v1.0/` (measurements under `results/v1.0/`) |
| `*.py` at the root | `src/`, `pipelines/`, `reporting/`, `checks/`, `tools/` |
| `session-transcript.md`, `session-transcript.json` | `docs/session/` |
| `MANIFEST-1.0.json`, `Assignment 1 Instructions.txt` | `releases/1.0/`, `docs/course/` |

View the report by opening `report.pdf`, and the website by serving the extracted `site/` over HTTP (`python -m http.server 8001 --directory site`) — opening `index.html` from the file system breaks the data fetches. The paths printed inside the report and the website are the archive paths above, because that is the layout the release shipped in.

The release-time copy of this file is inside the archive at `SUBMISSION.md`. It is frozen with the rest of the release and is not updated when this document changes.

## Verifying the archive

```powershell
python -m checks.v1_release
```

The check opens the archive read-only, verifies all 12,101 payloads against `releases/1.0/MANIFEST-1.0.json`, verifies the archive hash against `releases/1.0/Assignment1-submission.zip.sha256`, and rewrites `results/v1.0/package_check.json` from the verified bytes. Current values:

- members: 12,102 (12,101 payloads plus the manifest)
- size: 721.81 MiB (756,877,025 bytes)
- SHA-256: `cec084b56af1111e1b7a78f7f0d9581900cba985d86d40a01958437db856bb55`

`tools/package_submission.py` writes the same record at build time and refuses to overwrite a frozen release, so the record is refreshed by verification rather than by rebuilding.

## Relation to Versions 2.0 and 2.1

The 2.0 and 2.1 work is an extension beyond the assignment, not a replacement for it. Those releases stay in their own archives and are not inside the 1.0 ZIP:

- `releases/2.0/Assignment1-2.0.zip` — ANN, open-set and long-tail extension; contains a transcript of the 1.0-era session.
- `releases/2.1/Assignment1-2.1.zip` — audit-driven revision of 2.0; by design it publishes code, redacted manifests, aggregate metrics and figures only, and therefore contains **no** transcript, images, vectors or checkpoints.

If 2.0 or 2.1 is submitted alongside, say so explicitly: the 1.0 archive does not contain them.

## Session transcript

`docs/session/session-transcript.{md,json}` is a local, git-ignored export of the OpenCode session. It has been re-exported after the 2.0 and 2.1 work, so the working copy now covers the whole session rather than stopping at the 1.0 release. The 1.0-era bytes are not lost: they are hash-locked inside both the 1.0 and 2.0 archives.

The export includes local filesystem paths and machine-specific commands. Review it before submitting, then include `session-transcript.md` (and `session-transcript.json` if a full tool-level record is wanted) alongside the archive.

## Upload size

The archive is 721.81 MiB, dominated by derived website and measurement assets — `site/` is 444.59 MiB and `data/` is 342.93 MiB uncompressed, with `site/assets/data.json` alone at 132.41 MiB. Check the Blackboard upload limit before submitting. If the archive is too large:

1. Upload the ZIP to external storage or host the site separately, and submit the link with `report/v1.0/report.pdf`.
2. Submit a slim bundle of the root files only — report, code, protocol documents and transcript — which is about 36 MiB compressed across 35 members, and point to the hosted site for A5. A slim bundle is a derived convenience copy, not a release; it is not covered by the checks above, so regenerate and re-verify it if you build one.

Rebuilding the 1.0 archive to shrink it is not an option: it is a frozen release whose hash is recorded in `results/v1.0/package_check.json` and whose payloads are hash-locked in `MANIFEST-1.0.json`.

## Scope of the result

The report follows the assignment's five Basic Tasks (A1–A5) and three Advanced Tasks (B1–B3), followed by OpenCode decisions and reproduction details (C). Release notes: `releases/1.0/RELEASE-1.0.md`. The archive is a Version 1.0 record: the active source tree has since been reorganized for the Version 2.x work, so the reproduction commands in the current `README.md` are not the commands that built this archive. `archive/` holds dated snapshots of superseded work and no longer appears in any release. Public Hugging Face publication requires local authentication and is not claimed.
