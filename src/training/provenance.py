"""Small immutable artifact provenance helpers for the 2.1 revision."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def artifact(path, root):
    return {'path': Path(path).relative_to(root).as_posix(), 'sha256': sha256(path), 'bytes': Path(path).stat().st_size}


def assert_content_disjoint(rows):
    seen = {}
    for row in rows:
        digest = row['sha256']
        if digest in seen:
            raise AssertionError(f'Duplicate image content: {seen[digest]} and {row["image_id"]}')
        seen[digest] = row['image_id']
