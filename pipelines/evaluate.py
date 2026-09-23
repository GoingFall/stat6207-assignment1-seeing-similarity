"""Run the locked V3 experiment. All reference draws reuse image embeddings."""
from pathlib import Path
import itertools, json, hashlib, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import f1_score, confusion_matrix
from src.retrieval.encoders import encode
from src.retrieval.distances import normalize, pairwise
from pipelines.prepare_data import OUT, digest, canonical,R1


def load_locked():
    p=json.loads((OUT/'protocol.json').read_text()); rows=json.loads((OUT/'manifest.json').read_text())
    assert digest((OUT/'protocol.json').read_bytes())==(OUT/'protocol.sha256').read_text().strip()
    assert digest((OUT/'test_manifest.json').read_bytes())==p['test_manifest_sha256']
    assert digest((OUT/'manifest.json').read_bytes())==p['manifest_sha256']
    for r in rows: assert digest((OUT/r['image']).read_bytes())==r['processed_sha256']
    variants=json.loads((OUT/'variant_hashes.json').read_text())
    assert digest(canonical(variants))==p['variant_hashes_sha256']
    for path,sha in variants.items(): assert digest((OUT/path).read_bytes())==sha
    return p,rows


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    p,rows=load_locked(); protocol_hash=(OUT/'protocol.sha256').read_text().strip()
    out=R1;out.mkdir(exist_ok=True); cache=OUT/'embeddings';cache.mkdir(exist_ok=True)
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    refs=np.array([i for i,r in enumerate(rows) if r['split']=='reference'])
    label=np.array([r['source_label']=='dog' for r in rows]); truth=label[test]
    conditions=list(p['conditions']); nonclean=[c for c in conditions if c!='clean']
    ordered_paths=[OUT/r['image'] for r in rows]+[OUT/c/f"{rows[i]['id']}.png" for c in nonclean for i in test]
    all_metrics=[]; pred_cache={};score_cache={};norm_checks=[]
    for model in p['models']:
        path=cache/f'{model}.npy';meta=cache/f'{model}.json'
        if path.exists():
            assert json.loads(meta.read_text())['protocol_hash']==protocol_hash
            features=np.load(path)
        else:
            print('Encoding',model,len(ordered_paths),'inputs',flush=True)
            features,_=encode(model,ordered_paths,'cuda')
            np.save(path,features)
            meta.write_text(json.dumps(dict(protocol_hash=protocol_hash,torch=torch.__version__,device=torch.cuda.get_device_name(0),image_order='manifest then nonclean conditions then test manifest order',dimensions=features.shape[1]),indent=2))
        for condition in conditions:
            query=features[test] if condition=='clean' else features[len(rows)+nonclean.index(condition)*len(test):len(rows)+(nonclean.index(condition)+1)*len(test)]
            for mode in (['raw','unit'] if condition=='clean' else ['raw']):
                q=normalize(query) if mode=='unit' else query
                reference=normalize(features[refs]) if mode=='unit' else features[refs]
                unit_distances={}
                for metric in p['metrics']:
                    d=pairwise(q,reference,metric)
                    if mode=='unit':unit_distances[metric]=d
                    for size in ([10,25,50,100] if condition=='clean' and mode=='raw' else [100]):
                        ks=[1,3,5,7,9] if size==100 and condition=='clean' and metric=='cosine' and mode=='raw' else [5]
                        for k in ks:
                            key=f'{model}|{metric}|{mode}|{condition}|{size}|{k}'
                            preds=[];p5=[];metrics=[]
                            for seed in p['seeds']:
                                ids=np.array(p['subsets'][str(seed)][str(size)])
                                # Stable image ID tie break independent of shuffle order.
                                ids=np.sort(ids); columns=np.searchsorted(refs,ids)
                                ranking=np.argsort(d[:,columns],axis=1,kind='stable')
                                neighbors=label[ids][ranking[:,:k]]
                                pred=neighbors.mean(axis=1)>.5
                                precision=(label[ids][ranking[:,:5]]==truth[:,None]).mean(axis=1)
                                preds.append(pred);p5.append(precision)
                                metrics.append(dict(seed=seed,accuracy=float((pred==truth).mean()),macro_f1=float(f1_score(truth,pred,average='macro')),p5=float(precision.mean()),confusion=confusion_matrix(truth,pred,labels=[False,True]).tolist()))
                            pred_cache[key]=np.array(preds,dtype=np.uint8);score_cache[key]=np.array(p5,dtype=np.float32)
                            all_metrics.append(dict(key=key,model=model,metric=metric,mode=mode,condition=condition,size=size,k=k,seeds=metrics,
                                accuracy=float(np.mean([m['accuracy'] for m in metrics])),seed_sd=float(np.std([m['accuracy'] for m in metrics],ddof=1)),p5=float(np.mean(p5)),macro_f1=float(np.mean([m['macro_f1'] for m in metrics]))))
                if mode=='unit':
                    discrepancy=float(np.max(np.abs(unit_distances['l2']**2-2*unit_distances['cosine'])))
                    assert discrepancy<1e-5
                    norm_checks.append(dict(model=model,max_identity_error=discrepancy))
        print('Evaluated',model,flush=True)
    np.savez_compressed(out/'predictions.npz',**pred_cache);np.savez_compressed(out/'p5.npz',**score_cache)
    payload=dict(protocol_hash=protocol_hash,test_manifest_sha256=p['test_manifest_sha256'],test_ids=[rows[i]['id'] for i in test],truth=truth.astype(int).tolist(),metrics=all_metrics,normalization_checks=norm_checks)
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2))
    print('Formal evaluation complete',flush=True)


if __name__=='__main__':main()
