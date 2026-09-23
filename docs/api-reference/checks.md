# API Catalog: checks

由源码自动生成。检索结果只负责候选召回，是否复用必须继续阅读真实源码。

## `checks/browser_check.py`

### `checks.browser_check.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/browser_check.py:8`
- Signature: `main()`

## `checks/delivery_check.py`

### `checks.delivery_check.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/delivery_check.py:9`
- Signature: `main()`

## `checks/local_replay_preflight.py`

### `checks.local_replay_preflight._artifact_status`

- Kind: `function`
- Visibility: `private`
- Source: `checks/local_replay_preflight.py:18`
- Signature: `_artifact_status(path: Path, expected: str | None = None) -> dict`

Report a local artifact's presence and compare its digest when requested.

### `checks.local_replay_preflight.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/local_replay_preflight.py:34`
- Signature: `main() -> None`

Report local replay prerequisites and fail when required evidence is absent.

Returns:
    None. Emits the evidence summary as JSON on stdout.

## `checks/review_report.py`

### `checks.review_report.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/review_report.py:7`
- Signature: `main()`

## `checks/v1_release.py`

### `checks.v1_release.verify`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v1_release.py:24`
- Signature: `verify(archive = ARCHIVE, manifest = MANIFEST)`

Verify the archive against its manifest and sidecar, returning the packaging record.

Args:
    archive: Frozen ZIP under audit, opened read-only.
    manifest: Manifest whose ``files`` map records one payload per archive member.

Returns:
    The packaging record derived from the archive's own bytes.

Raises:
    AssertionError: If the sidecar hash, member set or any payload fails to match.

### `checks.v1_release.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v1_release.py:70`
- Signature: `main()`

Write a checked record for the frozen archive without rebuilding it.

Returns:
    None. Emits a JSON summary to stdout after the record is written.

## `checks/v21_contracts.py`

### `checks.v21_contracts.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_contracts.py:8`
- Signature: `main()`

## `checks/v21_exact_audit.py`

### `checks.v21_exact_audit.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_exact_audit.py:12`
- Signature: `main()`

## `checks/v21_release.py`

### `checks.v21_release._member_path`

- Kind: `function`
- Visibility: `private`
- Source: `checks/v21_release.py:11`
- Signature: `_member_path(name)`

Repository-relative path that an archive member has in the current layout.

### `checks.v21_release._zip_entry`

- Kind: `function`
- Visibility: `private`
- Source: `checks/v21_release.py:16`
- Signature: `_zip_entry(names, recorded)`

Archive member holding ``recorded``, in either the pre-migration or the current layout.

``resolve`` maps the legacy prefixes recorded inside the frozen 2.1 lock onto the current
layout and returns an already-migrated path unchanged, so one lookup covers both the
published archive (``data_v21/``) and a rebuild from the current tree (``data/v2.1/``).

Args:
    names: Member names of the archive under audit.
    recorded: Path as written inside the lock manifest.

Returns:
    The member name whose current-layout path equals the recorded one.

Raises:
    KeyError: If no member maps onto the recorded path.

### `checks.v21_release.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_release.py:40`
- Signature: `main()`

## `checks/v21_summarize_historical.py`

### `checks.v21_summarize_historical.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_summarize_historical.py:14`
- Signature: `main()`

## `checks/v21_verify.py`

### `checks.v21_verify.load`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_verify.py:17`
- Signature: `load(path)`

### `checks.v21_verify.check_artifact`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_verify.py:20`
- Signature: `check_artifact(a)`

### `checks.v21_verify.compare`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_verify.py:26`
- Signature: `compare(actual, expected, tolerance = 1e-07)`

### `checks.v21_verify.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v21_verify.py:37`
- Signature: `main()`

## `checks/v2_audit_sift1m_exact.py`

### `checks.v2_audit_sift1m_exact.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_audit_sift1m_exact.py:16`
- Signature: `main() -> None`

## `checks/v2_ci.py`

### `checks.v2_ci.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_ci.py:26`
- Signature: `main() -> None`

## `checks/v2_contracts.py`

### `checks.v2_contracts.sha256_file`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_contracts.py:11`
- Signature: `sha256_file(path: Path) -> str`

### `checks.v2_contracts.assert_no_self_match`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_contracts.py:19`
- Signature: `assert_no_self_match(query_ids: np.ndarray, result_ids: np.ndarray) -> None`

### `checks.v2_contracts.assert_product_disjoint`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_contracts.py:27`
- Signature: `assert_product_disjoint(train_product_ids: set[int], test_product_ids: set[int]) -> None`

### `checks.v2_contracts.verify_hash_manifest`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_contracts.py:33`
- Signature: `verify_hash_manifest(manifest_path: Path, root: Path) -> None`

## `checks/v2_review_artifacts.py`

### `checks.v2_review_artifacts.load`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:21`
- Signature: `load(p)`

### `checks.v2_review_artifacts.rows`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:22`
- Signature: `rows(p)`

### `checks.v2_review_artifacts.sha`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:23`
- Signature: `sha(p)`

### `checks.v2_review_artifacts.check`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:32`
- Signature: `check(t)`

### `checks.v2_review_artifacts.softmax`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:85`
- Signature: `softmax(z)`

### `checks.v2_review_artifacts.ece`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_artifacts.py:105`
- Signature: `ece(p, y)`

## `checks/v2_review_extra.py`

### `checks.v2_review_extra.load`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_extra.py:9`
- Signature: `load(p)`

### `checks.v2_review_extra.sha`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_review_extra.py:10`
- Signature: `sha(p)`

## `checks/v2_verify.py`

### `checks.v2_verify.jsonl`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_verify.py:15`
- Signature: `jsonl(path: Path) -> list[dict]`

### `checks.v2_verify.canonical_hash`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_verify.py:19`
- Signature: `canonical_hash(value: object) -> str`

### `checks.v2_verify.verify_sidecar`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_verify.py:23`
- Signature: `verify_sidecar(lock_path: Path) -> None`

### `checks.v2_verify.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/v2_verify.py:30`
- Signature: `main() -> None`

## `checks/verify.py`

### `checks.verify.main`

- Kind: `function`
- Visibility: `public`
- Source: `checks/verify.py:9`
- Signature: `main()`
