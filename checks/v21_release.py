"""Data-free public package audit, usable without any restricted artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path
from tools.package_v21 import permitted
from src.training.provenance import sha256


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
        for name,rec in manifest['files'].items():
            data=z.read(name)
            assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'],name
        lock=json.loads(z.read('data_v21/manifests/lock.json'))
        assert hashlib.sha256(z.read('configs/v21/experiment.json')).hexdigest()==lock['config_sha256']
        seen=set();ids=set()
        for split,rec in lock['manifests'].items():
            data=z.read(rec['path']);assert hashlib.sha256(data).hexdigest()==rec['sha256']
            for row in json.loads(data):
                assert set(row)=={'image_id','class_id','split','sha256','bytes','license'}
                assert row['sha256'] not in seen and row['image_id'] not in ids
                seen.add(row['sha256']);ids.add(row['image_id'])
        for rec in lock['long_tail']['runs']:
            assert hashlib.sha256(z.read(rec['path'])).hexdigest()==rec['sha256']
    print(json.dumps({'status':'passed','entries':len(names),'content_unique_images':len(seen),'zip_sha256':sha256(path)},indent=2))


if __name__=='__main__':main()
