"""Independent numerical, sampling, lock and exported retrieval checks."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from pipelines.evaluate import load_locked
from pipelines.prepare_data import OUT,ROOT

def main():
    p,rows=load_locked();test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);truth=np.array([rows[i]['source_label']=='dog' for i in test])
    assert len(test)==500 and truth.sum()==250 and len(rows)==1500
    assert json.loads((OUT/'test_manifest.json').read_text())==[rows[i] for i in test]
    assert len({r['duplicate_group'] for r in rows})==1500
    audit=json.loads((OUT/'audit.json').read_text());pilot={r['path'] for r in audit['records'] if r['pilot']}
    assert not pilot.intersection(r['path'] for r in rows)
    for seed in p['seeds']:
        prev=set()
        for size in [10,25,50,100]:
            ids=p['subsets'][str(seed)][str(size)];assert len(set(ids))==2*size and prev<=set(ids)
            assert all(rows[i]['split']=='reference' for i in ids)
            assert sum(rows[i]['source_label']=='cat' for i in ids)==size;prev=set(ids)
    e=json.loads((R1/'evaluation.json').read_text());pred=np.load(R1/'predictions.npz');ps=np.load(R1/'p5.npz')
    assert e['protocol_hash']==(OUT/'protocol.sha256').read_text().strip()
    for m in e['metrics']:
        a=pred[m['key']];assert a.shape==(10,500)
        assert np.isclose(m['accuracy'],(a==truth).mean())
        assert np.isclose(m['p5'],ps[m['key']].mean())
        assert np.isclose(m['macro_f1'],np.mean([f1_score(truth,v,average='macro') for v in a]))
        for si,s in enumerate(m['seeds']):assert np.isclose(s['accuracy'],(a[si]==truth).mean())
    site=json.loads((ROOT/'site/v1.0/assets/data.json').read_text());refs=np.array(sorted(p['subsets']['1001']['100']))
    reference_labels=np.array([rows[i]['source_label']=='dog' for i in refs]);nonclean=[c for c in p['conditions'] if c!='clean']
    checked=0;normalization=[]
    for model in p['models']:
        f=np.load(OUT/'embeddings'/f'{model}.npy');assert f.shape[0]==4500 and np.isfinite(f).all()
        for c in p['conditions']:
            q=f[test] if c=='clean' else f[1500+500*nonclean.index(c):2000+500*nonclean.index(c)]
            # Independent float64 check: normalize both sides, then use SciPy.
            x=q.astype(np.float64);y=f[refs].astype(np.float64)
            xn=np.linalg.norm(x,axis=1);yn=np.linalg.norm(y,axis=1)
            assert (xn>0).all() and (yn>0).all()
            cosine=cdist(x,y,'cosine');unit_l2=cdist(x/xn[:,None],y/yn[:,None],'euclidean')
            cr=np.argsort(cosine,axis=1,kind='stable');ur=np.argsort(unit_l2,axis=1,kind='stable')
            raw_rank=np.argsort(cdist(x,y,'euclidean'),axis=1,kind='stable')
            error=float(np.max(np.abs(unit_l2**2-2*cosine)))
            mismatches=int(np.any(cr!=ur,axis=1).sum())
            assert error<1e-12 and mismatches==0
            normalization.append(dict(model=model,condition=c,queries=len(x),references=len(y),unit_full_order_mismatches=mismatches,raw_top5_order_mismatches=int(np.any(raw_rank[:,:5]!=cr[:,:5],axis=1).sum()),max_identity_error=error,query_norm_range=[float(xn.min()),float(xn.max())],reference_norm_range=[float(yn.min()),float(yn.max())]))
            for metric,scipy_metric in [('l1','cityblock'),('l2','euclidean'),('cosine','cosine')]:
                d=cdist(q,f[refs],scipy_metric);rank=np.argsort(d,axis=1,kind='stable');key=f'{model}|{metric}|{c}'
                # Float32 arithmetic may reorder only numerically tied distances.
                exported=np.array(site['ranks'][key]);sd=np.take_along_axis(d,exported,axis=1)
                assert (np.diff(sd,axis=1)>=-5e-4).all()
                assert np.allclose(d,np.array(site['distances'][key]),atol=5e-4,rtol=1e-5)
                prediction=reference_labels[exported[:,:5]].mean(axis=1)>.5
                assert np.array_equal(prediction,pred[f'{model}|{metric}|raw|{c}|100|5'][0]);checked+=1
    assert len(e['comparisons'])==12
    geometry=dict(status='passed',protocol_hash=e['protocol_hash'],precision='float64',scope='seed 1001, 100 references/class; all 500 queries, three encoders and seven conditions',note='Ordered ranks, including all 200 references; raw top-five mismatches need not imply different sets or KNN predictions. Diagnostic only; frozen experiment uses float32.',configurations=normalization)
    serialized=json.dumps(geometry,indent=2)
    (R1/'normalization_check.json').write_text(serialized)
    (ROOT/'site/v1.0/assets/normalization_check.json').write_text(serialized)
    for family,n in [('models',3),('distances',9)]:
        group=sorted([r for r in e['comparisons'] if r['family']==family],key=lambda r:r['p']);assert len(group)==n
        expected=np.minimum(1,np.maximum.accumulate([r['p']*(n-i) for i,r in enumerate(group)]))
        assert np.allclose(expected,[r['holm_p'] for r in group])
    result=dict(status='passed',metric_records=len(e['metrics']),independent_distance_configurations=checked,checks=['all locked image hashes','balanced 1000/500 split','unique duplicate groups and pilot exclusion','nested balanced reference draws','all accuracy, macro-F1 and P@5 aggregates','SciPy distances for all 63 exported configurations','website rankings and votes match evaluation','both Holm families'])
    (R1/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
