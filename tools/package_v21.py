"""Allowlisted, verified revision release; never includes another release ZIP."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from src.training.provenance import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def permitted(name):
    from pathlib import PurePosixPath
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts: return False
    if any(x in p.parts for x in ('session','raw','processed','embeddings','__pycache__','.git')): return False
    if p.suffix in ('.npy','.npz','.pt','.pth','.faiss','.jpg','.jpeg','.zip'): return False
    if 'partial' in p.name or 'session-transcript' in p.name: return False
    if name in ('README.md','VERSION','requirements.txt','environment.yml','.gitignore','.gitattributes','MANIFEST-2.1.json'): return True
    if p.suffix=='.py' and p.parts[0] in ('src','pipelines','checks','reporting','tools'): return True
    if name.startswith('.github/workflows/') and p.suffix in ('.yml','.yaml'): return True
    if name.startswith(('configs/v21/','data_v21/manifests/','results_v21/')) and p.suffix=='.json': return True
    if name.startswith('results_v21/figures/') and p.suffix=='.png': return True
    return name in ('docs/report/V21-RESULTS.md','docs/report/V2-REVIEW-2026-09-23.md','releases/2.1/RELEASE-2.1.md')


def main():
    folder=ROOT/'releases/2.1'
    archive_path=folder/'Assignment1-2.1.zip'
    if archive_path.exists() or (folder/'MANIFEST-2.1.json').exists(): raise RuntimeError('Revision release already exists')
    if (ROOT/'VERSION').read_text().strip()!='2.1':raise RuntimeError('VERSION must be 2.1')
    subprocess.run([sys.executable,'-m','checks.v21_contracts'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','checks.v21_verify'],cwd=ROOT,check=True)
    write_json(ROOT/'results_v21/release_source_snapshot.json', {
        'scope':'Packaging-time source snapshot; pre-training critical source hashes are separately in provenance.json',
        'files':{p.relative_to(ROOT).as_posix():sha256(p) for directory in ('src','pipelines','checks','reporting','tools') for p in (ROOT/directory).rglob('*.py')},
    })
    paths=[]
    for directory in ('src','pipelines','checks','reporting','tools','configs/v21','data_v21/manifests','results_v21','.github/workflows'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and permitted(p.relative_to(ROOT).as_posix()))
    for name in ('README.md','VERSION','requirements.txt','environment.yml','.gitignore','.gitattributes','docs/report/V21-RESULTS.md','docs/report/V2-REVIEW-2026-09-23.md','releases/2.1/RELEASE-2.1.md'):
        p=ROOT/name
        if not p.is_file():raise FileNotFoundError(name)
        paths.append(p)
    records={p.relative_to(ROOT).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(set(paths))}
    manifest={'version':'2.1','scope':'Revised iNaturalist evaluation; SOP/SIFT V2 measurements are historical, not rebenchmarked','publication_boundary':'Code, redacted ID/content-hash manifests, aggregate metrics, environment/code provenance, figures. No images, vectors, models, predictions, transcripts or nested archives.','files':records}
    write_json(folder/'MANIFEST-2.1.json',manifest)
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in records:z.write(ROOT/name,name)
        z.write(folder/'MANIFEST-2.1.json','MANIFEST-2.1.json')
    (folder/'Assignment1-2.1.zip.sha256').write_text(sha256(archive_path)+'  Assignment1-2.1.zip\n')
    subprocess.run([sys.executable,'-m','checks.v21_release'],cwd=ROOT,check=True)


if __name__=='__main__':main()
