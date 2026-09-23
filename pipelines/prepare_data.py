"""V3 automatic audit, group-separated sampling and immutable test lock."""
from pathlib import Path
import hashlib, json, io, random
import numpy as np
import pandas as pd
import imagehash
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import kagglehub

from src.paths import OUT, R1, ROOT, V1
def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def canvas(path):
    with Image.open(path) as im: return ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC)
def main():
    if (OUT/'protocol.json').exists(): raise RuntimeError('Protocol already locked. Do not overwrite.')
    OUT.mkdir(exist_ok=True); (OUT/'images').mkdir(exist_ok=True)
    source=Path(kagglehub.dataset_download('bhavikjikadara/dog-and-cat-classification-dataset/versions/1'))
    pilot=pd.read_csv(ROOT/'data/v1.0/provenance/pilot_manifest.csv'); excluded=set(pilot[pilot.source.str.contains('bhavik')].source_path)
    paths=[]
    for label in ['cat','dog']:
        items=sorted(p for p in source.rglob('*.jpg') if p.parent.name.lower()==label)
        random.Random(42001+(label=='dog')).shuffle(items)
        paths.extend(items[:1800])
    paths=sorted(set(paths)|{source/p for p in excluded})
    records=[]; rejected=[]
    for p in paths:
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im:
                im=ImageOps.exif_transpose(im).convert('RGB'); im.load()
                pixel=digest(str(im.size).encode()+im.tobytes()); ph=str(imagehash.phash(im))
            records.append(dict(path=str(p.relative_to(source)),label=p.parent.name.lower(),sha=digest(p.read_bytes()),pixel=pixel,phash=ph,pilot=str(p.relative_to(source)) in excluded))
        except Exception as exc: rejected.append(dict(path=str(p.relative_to(source)),reason=str(exc)))
    n=len(records); parents=list(range(n))
    def find(a):
        while parents[a]!=a: parents[a]=parents[parents[a]]; a=parents[a]
        return a
    def union(a,b): parents[find(a)]=find(b)
    hashes=np.array([int(r['phash'],16) for r in records],dtype=np.uint64)
    pairs=[]; exact={}
    for i,r in enumerate(records):
        for field in ['sha','pixel']:
            key=(field,r[field])
            if key in exact: union(i,exact[key])
            else:exact[key]=i
        ds=np.bitwise_count(hashes[i+1:]^hashes[i])
        for j in np.flatnonzero(ds<=10)+i+1:
            union(i,int(j)); pairs.append(dict(a=i,b=int(j),hamming=int(ds[j-i-1])))
    # Supplemental geometric evidence on all perceptual-hash candidates; grouping remains conservative.
    orb=cv2.ORB_create(nfeatures=800); matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    for pair in pairs:
        aa=np.array(canvas(source/records[pair['a']]['path']).convert('L'))
        bb=np.array(canvas(source/records[pair['b']]['path']).convert('L'))
        ka,da=orb.detectAndCompute(aa,None); kb,db=orb.detectAndCompute(bb,None)
        good=[] if da is None or db is None or len(db)<2 else [m for m,nm in matcher.knnMatch(da,db,k=2) if m.distance<.75*nm.distance]
        pair['orb_matches']=len(good); pair['inliers']=0; pair['aligned_mae']=None
        if len(good)>=8:
            cv2.setRNGSeed(42001)
            h,mask=cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good]),cv2.RANSAC,3)
            if h is not None:
                pair['inliers']=int(mask.sum()); warped=cv2.warpPerspective(aa,h,(224,224)); valid=cv2.warpPerspective(np.ones_like(aa)*255,h,(224,224))>250
                if valid.any(): pair['aligned_mae']=float(np.abs(warped.astype(float)-bb)[valid].mean()/255)
    groups={}
    for i,r in enumerate(records):groups.setdefault(find(i),[]).append(i)
    representatives=[]
    for group,indices in groups.items():
        labels={records[i]['label'] for i in indices}
        conflict=len(labels)>1 and len({records[i]['pixel'] for i in indices})<len(indices)
        if any(records[i]['pilot'] for i in indices) or conflict:continue
        # At most one representative per conservative group removes group allocation ambiguities.
        idx=min(indices,key=lambda i:records[i]['path']); representatives.append((idx,group))
    selected=[]
    for label in ['cat','dog']:
        eligible=sorted([(i,g) for i,g in representatives if records[i]['label']==label],key=lambda t:records[t[0]]['path'])
        random.Random(42001+(label=='dog')).shuffle(eligible)
        assert len(eligible)>=750
        for k,(i,g) in enumerate(eligible[:750]):
            r=records[i]; ident=f'{label}_{k:04d}'; split='reference' if k<500 else 'test'
            image=canvas(source/r['path']); dest=OUT/'images'/f'{ident}.png'; image.save(dest)
            selected.append(dict(id=ident,source_label=label,split=split,path=r['path'],source_sha256=r['sha'],processed_sha256=digest(dest.read_bytes()),image=f'images/{ident}.png',duplicate_group=g,label_status='source_unverified',decode_status='passed'))
    selected=sorted(selected,key=lambda r:r['id']); test=[r for r in selected if r['split']=='test']
    lock=canonical(test)+b'\n'; (OUT/'test_manifest.json').write_bytes(lock)
    (OUT/'manifest.json').write_bytes(canonical(selected)+b'\n')
    audit=dict(method='pHash<=10 conservative connected groups; one representative/group; ORB audit supplementary',records=records,pairs=pairs,rejected=rejected,groups=[v for v in groups.values() if len(v)>1],selected_count=len(selected))
    (OUT/'audit.json').write_bytes(canonical(audit))
    subsets={}
    for seed in range(1001,1011):
        order={}
        for label in ['cat','dog']:
            ids=[i for i,r in enumerate(selected) if r['split']=='reference' and r['source_label']==label]
            random.Random(seed+(label=='dog')*10000).shuffle(ids); order[label]=ids
        subsets[str(seed)]={str(size):order['cat'][:size]+order['dog'][:size] for size in [10,25,50,100]}
    conditions={'clean':{},'blur_mild':{'radius':1},'blur_moderate':{'radius':2},'jpeg_mild':{'quality':50},'jpeg_moderate':{'quality':20},'occlusion_mild':{'side':71},'occlusion_moderate':{'side':112}}
    for condition,params in conditions.items():
        if condition=='clean':continue
        folder=OUT/condition;folder.mkdir(exist_ok=True)
        for r in test:
            image=Image.open(OUT/r['image']).convert('RGB')
            if condition.startswith('blur'):image=image.filter(ImageFilter.GaussianBlur(params['radius']))
            elif condition.startswith('jpeg'):
                buf=io.BytesIO();image.save(buf,format='JPEG',quality=params['quality'],subsampling=2,optimize=False);buf.seek(0);image=Image.open(buf).convert('RGB')
            else:
                rng=random.Random(int(digest(f"42004:{r['id']}".encode())[:16],16)); side=params['side'];x=int(rng.random()*(225-side));y=int(rng.random()*(225-side));ImageDraw.Draw(image).rectangle((x,y,x+side-1,y+side-1),fill=(127,127,127))
            image.save(folder/f"{r['id']}.png")
    variants={str(p.relative_to(OUT)):digest(p.read_bytes()) for c in conditions if c!='clean' for p in sorted((OUT/c).glob('*.png'))}
    (OUT/'variant_hashes.json').write_bytes(canonical(variants))
    display=[]
    for label in ['cat','dog']:
        ids=[r['id'] for r in test if r['source_label']==label];random.Random(42003).shuffle(ids);display+=ids[:5]
    protocol=dict(version=3,source='PetImages v1',test_manifest_sha256=digest(lock),manifest_sha256=digest((OUT/'manifest.json').read_bytes()),variant_hashes_sha256=digest(canonical(variants)),canvas='EXIF RGB ImageOps.fit 224 bicubic',conditions=conditions,subsets=subsets,k=5,k_sensitivity=[1,3,5,7,9],display_ids=display,seeds=list(range(1001,1011)),bootstrap=dict(seed=42005,repeats=5000),permutation=dict(seed=42006,repeats=10000),models=['resnet18','dinov2','clip'],metrics=['l1','l2','cosine'],audit_synsets=dict(cat=list(range(281,286)),dog=list(range(151,269))),audit_thresholds=dict(opposite=.8,source=.1),note='All labels source_unverified. No performance-based exclusion. Holm families: 3 cosine model pairs; 9 within-model distance pairs. Endpoint: mean accuracy over six nonclean conditions and ten reference seeds.')
    (OUT/'protocol.json').write_bytes(canonical(protocol)+b'\n');(OUT/'protocol.sha256').write_text(digest((OUT/'protocol.json').read_bytes()))
    print('LOCKED',len(selected),'images;',len(pairs),'near-duplicate candidate pairs;',len(rejected),'decode rejects; test hash',digest(lock),flush=True)

if __name__=='__main__':main()
