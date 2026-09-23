"""Extend checks.v2_review_artifacts evidence with drift and release checks."""
import sys,json,hashlib,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import faiss
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));faiss.omp_set_num_threads(16)
from src.monitoring.drift import drift_metrics,estimate_rbf_gamma
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
out=load('results_v2/checks/review-2026-09-23.json')
rows=[json.loads(l) for l in (ROOT/'data_v2/manifests/inat_birds_images.jsonl').read_text().splitlines()]
groups=defaultdict(list)
for r in rows:groups[r['sha256']].append({k:r[k] for k in ('image_id','class_id','split')})
out['inat_duplicate_details']=[{'sha256':h,'rows':rs} for h,rs in groups.items() if len({r['split'] for r in rs})>1]
out['config_hash_links']={}
for name in ('sop/encoding','sop/retrieval','ann/sift1m','inat/training'):
    r=load(f'results_v2/{name}.json')
    out['config_hash_links'][name]={'full_config_matches':r.get('config_sha256')==sha('configs/v2/experiment.json')}
    for field in ('sop_config','ann_config','exact_config','sift1m_config'):
        if field in r:out['config_hash_links'][name][field+'_matches']=r[field]==load('configs/v2/experiment.json')[field.replace('_config','')]
cfg=load('configs/v2/experiment.json')['monitoring'];stored=load('results_v2/inat/monitoring.json')
ref=np.load(ROOT/'data_v2/embeddings/inat_birds/validation_embeddings.npy')
rng=np.random.default_rng(cfg['seed']);proj=rng.standard_normal((cfg['random_projections'],384)).astype(np.float32);proj/=np.linalg.norm(proj,axis=1,keepdims=True)
gamma=estimate_rbf_gamma(ref,seed=cfg['seed']);null=[];half=len(ref)//2
for i in range(cfg['bootstrap_partitions']):
    p=rng.permutation(len(ref));null.append(drift_metrics(ref[p[:half]],ref[p[half:2*half]],proj,cfg['psi_bins'],cfg['rbf_mmd_samples'],cfg['seed']+i,rbf_gamma=gamma))
thresholds={k:float(np.quantile([r[k] for r in null],cfg['alert_quantile'])) for k in stored['thresholds']}
ev={}
for split in cfg['evaluation_splits']:
    x=np.load(ROOT/f'data_v2/embeddings/inat_birds/{split}_embeddings.npy');m=drift_metrics(ref,x,proj,cfg['psi_bins'],cfg['rbf_mmd_samples'],cfg['seed'],rbf_gamma=gamma)
    ev[split]={'metrics':m,'alerts':{k:m[k]>thresholds[k] for k in thresholds},'maximum_absolute_metric_delta':max(abs(m[k]-stored['evaluations'][split][k]) for k in m)}
out['monitoring_recomputed']={'thresholds':thresholds,'maximum_absolute_threshold_delta':max(abs(thresholds[k]-stored['thresholds'][k]) for k in thresholds),'evaluations':ev}
with zipfile.ZipFile(ROOT/'releases/2.0/Assignment1-2.0.zip') as z:
    s=z.read('docs/session/session-transcript.md');out['release']['transcript_bytes']=len(s)
    out['release']['scanned_transcript_markers']={x:s.count(x.encode()) for x in ('local_path','source_file_name','bird_train/','train_mini/','token','api_key')}
    out['release']['forbidden_artifacts']=[n for n in z.namelist() if Path(n).suffix.lower() in ('.jpg','.jpeg','.npy','.npz','.pt','.pth','.faiss') or n.startswith(('data_v2/raw/','data_v2/processed/','data_v2/embeddings/','releases/1.0/'))]
    for name in ('inat_birds_public_lock','inat_birds_images_public_lock'):
        p=f'data_v2/manifests/{name}.json';r=json.loads(z.read(p));h=hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==z.read(p.replace('.json','.sha256')).decode().split()[0]
    meta=json.loads(z.read('data_v2/manifests/inat_birds_public_lock.json'));im=json.loads(z.read('data_v2/manifests/inat_birds_images_public_lock.json'))
    assert im['metadata_lock_sha256']==hashlib.sha256(z.read('data_v2/manifests/inat_birds_public_lock.json')).hexdigest()
    for r in meta['manifests'].values():assert r['sha256']==hashlib.sha256(z.read(r['path'])).hexdigest()
    assert im['manifest_sha256']==hashlib.sha256(z.read(im['manifest'])).hexdigest()
    out['release']['public_lock_links_passed']=True
# Independently re-search 128 deterministic SOP queries against the full gallery.
x=np.load(ROOT/'data_v2/embeddings/sop/dinov2_test.npy');idx=faiss.IndexFlatIP(384);idx.add(x)
q=np.random.default_rng(6207).choice(len(x),128,replace=False);_,nn=idx.search(x[q],1001)
nn=np.array([r[r!=i][:1000] for i,r in zip(q,nn)]);saved=np.load(ROOT/'results_v2/sop/exact_top1000.npy',mmap_mode='r')[q]
out['sop_exact_research']={'queries':128,'gallery':len(x),'top1000_set_overlap':sum(len(set(a)&set(b)) for a,b in zip(nn,saved))/(128*1000),'top1_agreement':float(np.mean(nn[:,0]==saved[:,0]))}
(ROOT/'results_v2/checks/review-2026-09-23.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ('config_hash_links','monitoring_recomputed','release','sop_exact_research')},indent=2))
