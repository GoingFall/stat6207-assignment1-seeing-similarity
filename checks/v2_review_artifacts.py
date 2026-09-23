"""Read-only V2 experiment audit; writes only the dated review evidence JSON.

Run from the repository root, followed by checks.v2_review_extra.
Requires the original local datasets, embeddings and balanced checkpoint.
"""
import os, sys, json, hashlib, zipfile
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT))
from src.benchmarking.ann import sharded_exact_l2, choose_candidate
from src.training.metrics import classification_metrics, class_groups
from pipelines.v2_train_inat import aggregate_long_tail
from src.retrieval.open_set import select_threshold
faiss.omp_set_num_threads(16); torch.set_num_threads(16)
def load(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in (ROOT/p).read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
out={}
sop={s:rows(f'data_v2/manifests/sop_{s}.jsonl') for s in ('train','test')}
inat=rows('data_v2/manifests/inat_birds_images.jsonl')
tasks=[(ROOT/'data_v2/processed/sop'/r['path'],r['sha256'],r['bytes']) for rr in sop.values() for r in rr]+[(ROOT/r['local_path'],r['sha256'],r['bytes']) for r in inat]
def check(t):
    p,h,n=t
    return p.exists() and p.stat().st_size==n and sha(p)==h
with ThreadPoolExecutor(max_workers=8) as ex: good=sum(ex.map(check,tasks))
out['raw_image_hashes']={'checked':len(tasks),'passed':good}; assert good==len(tasks)
byhash=defaultdict(set)
for r in inat: byhash[r['sha256']].add(r['split'])
out['inat_cross_split_identical_bytes']=sum(len(s)>1 for s in byhash.values())
sets={s:{r['image_id'] for r in inat if r['split']==s} for s in {r['split'] for r in inat}}
assert sum(map(len,sets.values()))==len(set.union(*sets.values()))==len(inat)
for split,rr in sop.items():
    ids=load(f'data_v2/embeddings/sop/dinov2_{split}_ids.json')
    assert ids==[r['image_id'] for r in rr]
out['sop_id_alignment']=True
lab=np.array([r['product_id'] for r in sop['test']]); counts=Counter(lab)
neigh=np.load(ROOT/'results_v2/sop/exact_top1000.npy',mmap_mode='r')
hits={k:0 for k in (1,10,100)}; aps=0.; invalid=0; duplicates=0
for start in range(0,len(lab),256):
    nn=np.asarray(neigh[start:start+256]); q=np.arange(start,start+len(nn))
    assert ((nn>=0)&(nn<len(lab))).all() and not (nn==q[:,None]).any()
    duplicates+=sum(len(set(r))!=1000 for r in nn)
    rel=lab[nn]==lab[q,None]
    for k in hits: hits[k]+=np.any(rel[:,:k],axis=1).sum()
    den=np.array([min(counts[l]-1,1000) for l in lab[q]]); assert (den>0).all()
    aps+=np.sum(np.sum(np.cumsum(rel,axis=1)/np.arange(1,1001)*rel,axis=1)/den)
out['sop_metrics']={**{f'recall_at_{k}':float(v/len(lab)) for k,v in hits.items()},'map_at_1000':float(aps/len(lab)),'duplicate_rows':duplicates}
for k,v in load('results_v2/sop/retrieval.json')['standard_semantic'].items(): assert abs(out['sop_metrics'][k]-v)<1e-12
out['ann_selection']={}
for file in ('results_v2/sop/retrieval.json','results_v2/ann/sift1m.json'):
    r=load(file)
    out['ann_selection'][file]={}
    for family,f in r['families'].items():
        selected=choose_candidate(f['development_candidates'],.95)
        assert selected['name']==f['selected']
        out['ann_selection'][file][family]=f['test']['recall_at_10']
lt=load('data_v2/manifests/long_tail/index.json')
out['long_tail_sampling']=[]
for rec in lt['runs']:
    r=load(rec['path']); selected=[i for ids in r['selected_image_ids'].values() for i in ids]
    assert len(selected)==len(set(selected)) and set(selected)<=sets['train']
    assert all(len(r['selected_image_ids'][c])==n for c,n in r['counts'].items())
    assert max(r['counts'].values())/min(r['counts'].values())==10
    out['long_tail_sampling'].append({'seed':r['seed'],'images':len(selected),'min':min(r['counts'].values()),'max':max(r['counts'].values())})
training=load('results_v2/inat/training.json')
summary=aggregate_long_tail(training['long_tail'],list(training['long_tail_summary']))
assert summary==training['long_tail_summary']; out['long_tail_aggregate_matches']=True
emb=ROOT/'data_v2/embeddings/inat_birds'
train_y=np.load(emb/'train_labels.npy'); classes=np.unique(train_y)
train_y=np.searchsorted(classes,train_y); y=np.searchsorted(classes,np.load(emb/'test_labels.npy'))
x=np.load(emb/'test_embeddings.npy'); train_x=np.load(emb/'train_embeddings.npy')
model=torch.load(ROOT/training['balanced_probe']['model_path'],map_location='cpu',weights_only=False)
w=model['state_dict']['weight'].numpy(); b=model['state_dict']['bias'].numpy()
logits=x@w.T+b
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
probs=softmax(logits); m=classification_metrics(y,probs,dict(Counter(map(int,train_y))))
out['balanced_probe_recomputed']=m
idx=faiss.IndexFlatIP(384);idx.add(train_x); _,nn=idx.search(x,10)
p=np.zeros((len(y),len(classes)),np.float32)
for i,r in enumerate(train_y[nn]):
    for c,n in Counter(r).items():p[i,c]=n/10
out['knn_recomputed']=classification_metrics(y,p,dict(Counter(map(int,train_y))))
out['group_f1_correction_balanced_probe']={g:{'saved':m['groups'][g]['macro_f1'],'corrected':float(f1_score(y,probs.argmax(1),labels=sorted(cs),average='macro',zero_division=0))} for g,cs in class_groups(dict(Counter(map(int,train_y)))).items()}
osr=load('results_v2/inat/open_set.json'); scores=np.load(ROOT/osr['scores_path'])
out['open_set_recomputed']={}
for name,r in osr['methods'].items():
    kd,ud,kt,ut=[scores[f'{name}_{s}'] for s in ('known_development','unknown_development','known_test','unknown_test')]
    th=select_threshold(kd,ud,.95); assert th==r['threshold']
    yl=np.r_[np.zeros(len(kt)),np.ones(len(ut))]; ss=-np.r_[kt,ut]
    fpr,tpr,_=roc_curve(yl,ss)
    metrics={'auroc_ood':roc_auc_score(yl,ss),'aupr_ood':average_precision_score(yl,ss),'fpr_at_95_tpr_ood':fpr[np.flatnonzero(tpr>=.95)[0]],'known_coverage':np.mean(kt>=th),'unknown_rejection_rate':np.mean(ut<th)}
    assert all(abs(v-r['test'][k])<1e-12 for k,v in metrics.items())
    out['open_set_recomputed'][name]=metrics
def ece(p,y):
    conf=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((conf*15).astype(int),14)
    return sum(abs(np.sum(correct[bins==i])-np.sum(conf[bins==i])) for i in range(15))/len(y)
out['calibration_test_recomputed']={'ece_before':ece(probs,y),'ece_after':ece(softmax(logits/osr['calibration']['temperature']),y)}
# Demonstrate boundary tie failure using equal-distance vectors and different shards.
tie=[]
db=np.zeros((20,2),np.float32);q=np.zeros((1,2),np.float32)
for shard in (3,7,20):
    _,ids=sharded_exact_l2(db,q,5,1,shard);tie.append({'shard':shard,'ids':ids.tolist()})
out['exact_tie_reproducer']=tie
zpath=ROOT/'releases/2.0/Assignment1-2.0.zip'
with zipfile.ZipFile(zpath) as z:
    manifest=json.loads(z.read('MANIFEST-2.0.json')); records=manifest['files']|manifest['generated_public_files']
    assert set(z.namelist())==set(records)|{'MANIFEST-2.0.json'}
    assert all(len(z.read(n))==r['bytes'] and hashlib.sha256(z.read(n)).hexdigest()==r['sha256'] for n,r in records.items())
    out['release']={'sha256':sha(zpath),'entries':len(z.namelist()),'all_entry_hashes_passed':True,'session_files':[n for n in z.namelist() if 'session' in n]}
out['v1_archive_sha256']=sha(ROOT/'releases/1.0/Assignment1-submission.zip')
Path('results_v2/checks/review-2026-09-23.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
