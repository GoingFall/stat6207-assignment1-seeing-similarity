"""Package the current deliverables; frozen release archives are never overwritten."""
import json,zipfile,hashlib
from pipelines.prepare_data import ROOT,R1

def main():
    version=(ROOT/'VERSION').read_text().strip()
    label='v'+'.'.join(version.split('.')[:2])
    name='Assignment1-submission.zip' if version=='1.0' else f'Assignment1-{version}.zip'
    archive=ROOT/'releases'/version/name
    if archive.exists():raise RuntimeError(f'{archive.relative_to(ROOT).as_posix()} is a frozen release; refusing to overwrite.')
    for name in ['verification.json','browser_check.json','delivery_check.json']:
        assert json.loads((R1/name).read_text())['status']=='passed'
    assert json.loads((R1/'report_review/visual_review.json').read_text())['status']=='passed'
    assert json.loads((R1/'heatmap_check.json').read_text())['status']=='passed'
    assert json.loads((R1/'normalization_check.json').read_text())['status']=='passed'
    assert json.loads((R1/'supplement_check.json').read_text())['status']=='passed'
    files=[p for p in ROOT.iterdir() if p.is_file() and p.name in ['README.md','VERSION','requirements.txt','environment.yml']]
    for directory in ['src','pipelines','reporting','checks','tools','docs',f'data/{label}',f'results/{label}',f'site/{label}']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.name!='package_check.json' and p.suffix not in ['.log','.pyc','.tmp'] and '__pycache__' not in p.parts)
    files.append(ROOT/'report'/label/'report.pdf')
    files.sort(key=lambda p:p.relative_to(ROOT).as_posix())
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    manifest_path=archive.parent/f'MANIFEST-{version}.json'
    manifest_path.write_text(json.dumps(dict(version=version,files=manifest),indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.write(manifest_path,manifest_path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(member in z.namelist() for member in [f'report/{label}/report.pdf',f'site/{label}/index.html',f'data/{label}/protocol.json'])
        assert not any(member.startswith(('archive/','releases/','formal/','formal_site/')) for member in z.namelist())
        for member,record in manifest.items():assert hashlib.sha256(z.read(member)).hexdigest()==record['sha256']
        count=len(z.namelist())
    result=dict(archive=archive.relative_to(ROOT).as_posix(),size_mib=round(archive.stat().st_size/2**20,2),members=count,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),integrity='passed')
    (R1/'package_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (archive.parent/f'{archive.name}.sha256').write_text(result['sha256']+'  '+archive.name+'\n')

if __name__=='__main__':main()
