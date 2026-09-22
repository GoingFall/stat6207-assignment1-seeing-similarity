"""Package only current deliverables; historical versions remain in backup/."""
import json,zipfile,hashlib
from pipelines.prepare_data import ROOT,OUT

def main():
    for name in ['verification.json','browser_check.json','delivery_check.json']:
        assert json.loads((OUT/'results'/name).read_text())['status']=='passed'
    assert json.loads((OUT/'results/report_review/visual_review.json').read_text())['status']=='passed'
    assert json.loads((OUT/'results/heatmap_check.json').read_text())['status']=='passed'
    assert json.loads((OUT/'results/normalization_check.json').read_text())['status']=='passed'
    assert json.loads((OUT/'results/supplement_check.json').read_text())['status']=='passed'
    version=(ROOT/'VERSION').read_text().strip()
    archive=ROOT/('Assignment1-submission.zip' if version=='1.0' else f'Assignment1-{version}.zip')
    files=[]
    for p in ROOT.iterdir():
        if p.is_file() and p.name in ['README.md','VERSION','requirements.txt','environment.yml','report.pdf']:files.append(p)
    for directory in ['src','pipelines','reporting','checks','tools','docs','data','site']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.name!='package_check.json' and p.suffix not in ['.log','.pyc','.tmp'] and '__pycache__' not in p.parts)
    files.sort(key=lambda p:p.relative_to(ROOT).as_posix())
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    manifest_path=ROOT/f'MANIFEST-{version}.json'
    manifest_path.write_text(json.dumps(dict(version=version,files=manifest),indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.write(manifest_path,manifest_path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(name in z.namelist() for name in ['report.pdf','site/index.html','data/protocol.json'])
        assert not any(name.startswith(('backup/','formal/','formal_site/')) for name in z.namelist())
        for name,record in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
        count=len(z.namelist())
    result=dict(archive=archive.name,size_mib=round(archive.stat().st_size/2**20,2),members=count,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),integrity='passed')
    (OUT/'results/package_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (ROOT/f'{archive.name}.sha256').write_text(result['sha256']+'  '+archive.name+'\n')

if __name__=='__main__':main()
