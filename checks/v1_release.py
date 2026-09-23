"""Re-derive the Version 1.0 packaging record from the frozen archive itself.

``tools/package_submission.py`` writes ``results/v1.0/package_check.json`` at build time and
refuses to overwrite a frozen release, so the record cannot be refreshed by rebuilding. This
check instead reads the delivered archive, verifies every payload against
``releases/1.0/MANIFEST-1.0.json`` and the archive hash against its ``.sha256`` sidecar, and
rewrites the record with the verified values. A record that matches the bytes on disk is the
only thing it can produce: the archive is opened read-only and is never modified.
"""

import hashlib
import json
import sys
import zipfile

from src.paths import ROOT
from src.training.provenance import sha256, write_json

ARCHIVE = ROOT / "releases/1.0/Assignment1-submission.zip"
MANIFEST = ROOT / "releases/1.0/MANIFEST-1.0.json"
RECORD = ROOT / "results/v1.0/package_check.json"


def verify(archive=ARCHIVE, manifest=MANIFEST):
    """Verify the archive against its manifest and sidecar, returning the packaging record.

    Args:
        archive: Frozen ZIP under audit, opened read-only.
        manifest: Manifest whose ``files`` map records one payload per archive member.

    Returns:
        The packaging record derived from the archive's own bytes.

    Raises:
        AssertionError: If the sidecar hash, member set or any payload fails to match.
    """
    digest = sha256(archive)
    sidecar = archive.with_suffix(".zip.sha256").read_text().split()[0]
    if digest != sidecar:
        raise AssertionError(f"{archive.name} does not match its .sha256 sidecar")
    payloads = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise AssertionError("archive CRC validation failed")
        members = z.namelist()
        if len(members) != len(set(members)):
            raise AssertionError("archive stores a member twice")
        if set(members) != set(payloads) | {manifest.name}:
            raise AssertionError("member set differs from the manifest")
        if z.read(manifest.name) != manifest.read_bytes():
            raise AssertionError("embedded manifest differs from the release manifest")
        for name, record in payloads.items():
            data = z.read(name)
            if (
                len(data) != record["bytes"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                raise AssertionError(f"payload differs from the manifest: {name}")
    return dict(
        archive=archive.relative_to(ROOT).as_posix(),
        size_mib=round(archive.stat().st_size / 2**20, 2),
        members=len(payloads) + 1,
        payloads=len(payloads),
        sha256=digest,
        integrity="passed",
        verified_by="checks/v1_release.py",
    )


def main():
    """Write a checked record for the frozen archive without rebuilding it.

    Returns:
        None. Emits a JSON summary to stdout after the record is written.
    """
    record = verify()
    write_json(RECORD, record)
    sys.stdout.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
