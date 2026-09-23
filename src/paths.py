"""Single source of truth for the versioned project layout.

Directory convention
--------------------
``data/vMAJOR.MINOR/``     locked inputs, manifests and embeddings
``results/vMAJOR.MINOR/``  measurements, figures and audit outputs
``configs/vMAJOR.MINOR/``  preregistered parameters
``site/vMAJOR.MINOR/``     generated static site for that version
``report/vMAJOR.MINOR/``   generated report for that version
``releases/<VERSION>/``    immutable release archives, never renamed
``archive/<date>-<slug>/`` dated snapshots of superseded work

Frozen artifacts
----------------
Manifests, results and release archives are hash-locked and are never edited.
Those recorded *before* the layout migration therefore still store the old
ROOT-relative prefixes (``data_v2/``, ``results_v21/``, ``site/``, ``backup/``).
``resolve`` maps such a recorded path onto the current layout, so historical
evidence keeps its original bytes while the working tree keeps a single naming
convention. A recorded path that already names a current-layout root is passed
through to ``ROOT`` as written, so it is not migrated a second time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
CONFIGS = ROOT / 'configs'
SITE = ROOT/'site/v1.0'
REPORT = ROOT / 'report'
RELEASES = ROOT / 'releases'
ARCHIVE = ROOT / 'archive'

# Versioned roots.
V1, V2, V21 = DATA / 'v1.0', DATA / 'v2.0', DATA / 'v2.1'
R1, R2, R21 = RESULTS / 'v1.0', RESULTS / 'v2.0', RESULTS / 'v2.1'
C2, C21 = CONFIGS / 'v2.0', CONFIGS / 'v2.1'

# Version 1.0 kept its inputs and derived outputs under one root.
OUT = V1

# Roots of the current layout. A recorded path that already starts with one of these is returned
# unchanged, which is what keeps ``resolve`` idempotent. Every value in LEGACY_PREFIXES must appear
# here, otherwise resolving an already-migrated path would migrate it a second time.
CURRENT_PREFIXES = (
    'data/v1.0/', 'data/v2.0/', 'data/v2.1/',
    'results/v1.0/', 'results/v2.0/', 'results/v2.1/',
    'configs/v2.0/', 'configs/v2.1/',
    'site/v1.0/', 'report/v1.0/', 'archive/', 'releases/',
)

# Prefixes recorded inside hash-locked artifacts, mapped to the current layout.
# Most specific first: ``data/results/`` must precede ``data/``, and the dated ``backup/<snapshot>/``
# entries must precede the bare ``backup/`` entry, or the generic rule would win. The bare ``data/``
# entry stays last because it is the least specific of all.
LEGACY_PREFIXES = {
    'data/results/': 'results/v1.0/',
    'data_v2/': 'data/v2.0/',
    'data_v21/': 'data/v2.1/',
    'results_v2/': 'results/v2.0/',
    'results_v21/': 'results/v2.1/',
    'configs/v2/': 'configs/v2.0/',
    'configs/v21/': 'configs/v2.1/',
    'backup/pilot/': 'archive/2026-09-21-pilot-270-image/',
    'backup/v3-before-reorganization/': 'archive/2026-09-21-v1.0-before-source-reorganization/',
    'backup/before-normalization-clarification/': 'archive/2026-09-21-v1.0-before-normalization-clarification/',
    'backup/website-before-interaction-update/': 'archive/2026-09-21-v1.0-website-before-interaction-update/',
    'backup/': 'archive/',
    'site/': 'site/v1.0/',
    'data/': 'data/v1.0/',
}

# Whole recorded names that moved into a versioned directory. Matched exactly, so ``report.pdf.bak``
# is not treated as ``report.pdf``.
LEGACY_FILES = {
    'report.pdf': 'report/v1.0/report.pdf',
}


def resolve(relative) -> Path:
    """Return the absolute path for a ROOT-relative path recorded in an artifact.

    A recorded path that already names a current-layout root is joined to ``ROOT`` as written,
    so resolving the same recorded path twice lands on the same file.
    """
    text = str(relative).replace('\\', '/').strip()
    while text.startswith('./'):
        text = text[2:]
    if not text or any(text == prefix.rstrip('/') or text.startswith(prefix) for prefix in CURRENT_PREFIXES):
        return ROOT / text
    if text in LEGACY_FILES:
        return ROOT / LEGACY_FILES[text]
    for old, new in LEGACY_PREFIXES.items():
        if text == old.rstrip('/') or text.startswith(old):
            return ROOT / (new + text[len(old):])
    return ROOT / text
