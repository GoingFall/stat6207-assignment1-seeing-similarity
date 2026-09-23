"""Data-free public package audit, usable without any restricted artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path
from tools.package_v21 import permitted
from src.paths import ROOT,resolve
from src.training.provenance import sha256


def _member_path(name):
    """Repository-relative path that an archive member has in the current layout."""
    return resolve(name).relative_to(ROOT).as_posix()


def _zip_entry(names,recorded):
    """Archive member holding ``recorded``, in either the pre-migration or the current layout.

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
    """
    target=_member_path(recorded)
    for name in names:
        if _member_path(name)==target:
            return name
    raise KeyError(f'{recorded} (current path {target}) is absent from the archive')


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'releases/2.1/Assignment1-2.1.zip'
    assert sha256(path)==path.with_suffix('.zip.sha256').read_text().split()[0]
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('MANIFEST-2.1.json'))
        names=z.namelist()
        assert len(names)==len(set(names))
        assert set(names)==set(manifest['files'])|{'MANIFEST-2.1.json'}
        assert all(permitted(n) for n in names)
        # Two members that map onto one current path would make the lookup below ambiguous and
        # let a duplicate artifact pass the content checks.
        assert len({_member_path(n) for n in names})==len(names),'archive stores one artifact twice'
        for name,rec in manifest['files'].items():
            data=z.read(name)
            assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'],name
        # The frozen 2.1 archive stores the pre-migration layout, a rebuild stores the current one.
        # Both are read through the same lookup so neither layout is audited with weaker checks.
        lock=json.loads(z.read(_zip_entry(names,'data/v2.1/manifests/lock.json')))
        config=z.read(_zip_entry(names,'configs/v2.1/experiment.json'))
        assert hashlib.sha256(config).hexdigest()==lock['config_sha256']
        seen=set();ids=set()
        for split,rec in lock['manifests'].items():
            data=z.read(_zip_entry(names,rec['path']))
            assert hashlib.sha256(data).hexdigest()==rec['sha256']
            for row in json.loads(data):
                assert set(row)=={'image_id','class_id','split','sha256','bytes','license'}
                assert row['sha256'] not in seen and row['image_id'] not in ids
                seen.add(row['sha256']);ids.add(row['image_id'])
        for rec in lock['long_tail']['runs']:
            data=z.read(_zip_entry(names,rec['path']))
            assert hashlib.sha256(data).hexdigest()==rec['sha256']
    print(json.dumps({'status':'passed','entries':len(names),'content_unique_images':len(seen),'zip_sha256':sha256(path)},indent=2))


if __name__=='__main__':main()
