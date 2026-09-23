"""Check derived pixels, PDF layout, and supplementary site resources."""
import io,json,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import pymupdf
from pipelines.prepare_data import OUT,R1,ROOT,digest

def main():
    p=json.loads((OUT/'protocol.json').read_text());test=json.loads((OUT/'test_manifest.json').read_text());checked=0
    for r in test:
        clean=Image.open(OUT/r['image']).convert('RGB')
        for c,param in p['conditions'].items():
            if c=='clean':continue
            image=clean.copy()
            if c.startswith('blur'):image=image.filter(ImageFilter.GaussianBlur(param['radius']))
            elif c.startswith('jpeg'):
                b=io.BytesIO();image.save(b,format='JPEG',quality=param['quality'],subsampling=2,optimize=False);b.seek(0);image=Image.open(b).convert('RGB')
            else:
                rng=random.Random(int(digest(f"42004:{r['id']}".encode())[:16],16));s=param['side'];x=int(rng.random()*(225-s));y=int(rng.random()*(225-s));ImageDraw.Draw(image).rectangle((x,y,x+s-1,y+s-1),fill=(127,127,127))
            actual=Image.open(OUT/c/f"{r['id']}.png");assert actual.size==(224,224) and np.array_equal(np.array(image),np.array(actual));checked+=1
    doc=pymupdf.open(ROOT/'report/v1.0/report.pdf')
    for page in doc:
        for block in page.get_text('blocks'):
            assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width+1 and block[3]<=page.rect.height+1
    assert (ROOT/'report/v1.0/report.pdf').read_bytes()==(ROOT/'site/v1.0/report.pdf').read_bytes()
    ood=json.loads((R1/'ood/results.json').read_text());assert len(ood['images'])==20
    for model,items in ood['models'].items():
        assert len(items)==20
        for r in items:
            assert (ROOT/'site/v1.0/ood'/f"{r['id']}.png").exists()
            assert len(r['nearest'])==5
            for ident in r['nearest']:assert (ROOT/'site/v1.0/images'/f'{ident}.png').exists()
    meta=json.loads((ROOT/'site/v1.0/heatmaps/index.json').read_text())
    assert meta['protocol_hash']==(OUT/'protocol.sha256').read_text().strip()
    for record in meta['files']:
        raw=(ROOT/'site/v1.0/heatmaps'/f"{record['condition']}-{record['metric']}.bin").read_bytes()
        assert len(raw)==500*10*2*7*7 and digest(raw)==record['sha256']
    assert len(meta['files'])==21
    for r in test:
        for condition in p['conditions']:
            with Image.open(ROOT/'site/v1.0/heatmaps/views'/condition/f"{r['id']}.jpg") as image:assert image.size==(224,224)
    result=dict(status='passed',independently_regenerated_variants=checked,pdf_pages=len(doc),pdf_text_bounds='passed',ood_queries=20,ood_models=3,heatmap_files=21,heatmaps=210000,query_crop_images=3500)
    (R1/'delivery_check.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
