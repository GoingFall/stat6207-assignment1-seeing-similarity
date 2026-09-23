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
ROOT-relative prefixes (``data/v2.0/``, ``results/v2.1/``). ``resolve`` maps such a
recorded path onto the current layout, so historical evidence keeps its original
bytes while the working tree keeps a single naming convention.
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

# Prefixes recorded inside hash-locked artifacts, mapped to the current layout.
LEGACY_PREFIXES = {
    'data_v2/': 'data/v2.0/',
    'data_v21/': 'data/v2.1/',
    'results_v2/': 'results/v2.0/',
    'results_v21/': 'results/v2.1/',
    'configs/v2/': 'configs/v2.0/',
    'configs/v21/': 'configs/v2.1/',
    'backup/': 'archive/',
    'site/': 'site/v1.0/',
    'report.pdf': 'report/v1.0/report.pdf',
}


def resolve(relative) -> Path:
    """Return the absolute path for a ROOT-relative path recorded in an artifact."""
    text = str(relative).replace('\\', '/')
    for old, new in LEGACY_PREFIXES.items():
        if text == old.rstrip('/') or text.startswith(old):
            return ROOT / (new + text[len(old):])
    return ROOT / text
