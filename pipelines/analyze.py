"""Paired query bootstrap and prespecified Holm families, no winner selection."""
import itertools,json
import numpy as np
from pipelines.prepare_data import OUT,R1


def main():
    p=json.loads((OUT/'protocol.json').read_text()); out=R1
    data=json.loads((out/'evaluation.json').read_text());truth=np.array(data['truth'])
    pred=np.load(out/'predictions.npz');precision=np.load(out/'p5.npz')
    rng=np.random.default_rng(42005);cats=np.flatnonzero(truth==0);dogs=np.flatnonzero(truth==1)
    sampled=np.concatenate([rng.choice(cats,(5000,len(cats))),rng.choice(dogs,(5000,len(dogs)))],axis=1)
    weights=np.zeros((5000,len(truth)),dtype=np.float32)
    for row,idx in enumerate(sampled):weights[row]=np.bincount(idx,minlength=len(truth))/len(truth)
    scores=[];p5scores=[]
    for m in data['metrics']:
        scores.append((pred[m['key']]==truth).mean(axis=0));p5scores.append(precision[m['key']].mean(axis=0))
    scores=np.array(scores);p5scores=np.array(p5scores)
    boot=weights@scores.T;bootp=weights@p5scores.T
    for i,m in enumerate(data['metrics']):
        m['accuracy_ci']=np.quantile(boot[:,i],[.025,.975]).tolist();m['p5_ci']=np.quantile(bootp[:,i],[.025,.975]).tolist()
        m['bootstrap_degenerate']=bool(np.ptp(boot[:,i])==0)
        # Recompute macro-F1 per reference seed for every query bootstrap sample.
        prediction=pred[m['key']]
        tp=weights@((prediction==1)&(truth==1)).T;fp=weights@((prediction==1)&(truth==0)).T
        fn=weights@((prediction==0)&(truth==1)).T;tn=weights@((prediction==0)&(truth==0)).T
        f1=(2*tp/np.maximum(2*tp+fp+fn,1e-12)+2*tn/np.maximum(2*tn+fp+fn,1e-12))/2
        m['macro_f1_ci']=np.quantile(f1.mean(axis=1),[.025,.975]).tolist()
    nonclean=[c for c in p['conditions'] if c!='clean']; endpoints={}
    for model in p['models']:
        for metric in p['metrics']:
            endpoints[f'{model}|{metric}']=np.mean([(pred[f'{model}|{metric}|raw|{c}|100|5']==truth).mean(axis=0) for c in nonclean],axis=0)
    comparisons=[]
    for a,b in itertools.combinations(p['models'],2):comparisons.append(dict(family='models',a=f'{a}|cosine',b=f'{b}|cosine'))
    for model in p['models']:
        for a,b in itertools.combinations(p['metrics'],2):comparisons.append(dict(family='distances',a=f'{model}|{a}',b=f'{model}|{b}'))
    rng=np.random.default_rng(42006);signs=rng.choice(np.array([-1,1],dtype=np.int8),(10000,len(truth)))
    for r in comparisons:
        diff=endpoints[r['a']]-endpoints[r['b']];obs=float(diff.mean());null=signs@diff/len(diff)
        r.update(difference=obs,ci=np.quantile(weights@diff,[.025,.975]).tolist(),p=float((1+(np.abs(null)>=abs(obs)-1e-12).sum())/10001))
    for family in ['models','distances']:
        group=sorted([r for r in comparisons if r['family']==family],key=lambda r:r['p']);prev=0
        for i,r in enumerate(group):prev=max(prev,min(1,r['p']*(len(group)-i)));r['holm_p']=prev
    data['comparisons']=comparisons;data['endpoint_means']={k:float(v.mean()) for k,v in endpoints.items()}
    data['interval_note']='95% stratified paired query bootstrap; fixed reference pool and ten realized draws. Seed SD separately reported. Degenerate intervals do not prove perfect population performance.'
    (out/'evaluation.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(endpoint_means=data['endpoint_means'],comparisons=comparisons),indent=2),flush=True)


if __name__=='__main__':main()
