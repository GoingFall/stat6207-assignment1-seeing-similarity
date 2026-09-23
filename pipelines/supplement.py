"""Post-hoc descriptive pixel baseline and class-error audit on locked data."""
import json
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from pipelines.evaluate import load_locked
from pipelines.prepare_data import R1,ROOT,OUT

def main():
    p,rows=load_locked();torch.set_num_threads(4)
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    truth=np.array([rows[i]['source_label']=='dog' for i in test])
    pixels=[]
    for r in rows:
        with Image.open(OUT/r['image']) as im:pixels.append(np.asarray(im.convert('RGB'),dtype=np.float32).reshape(-1)/255.)
    x=torch.tensor(np.stack(pixels),device='cuda',dtype=torch.float64)
    # Shared 224x224 RGB canvas, no model processor or feature normalization.
    records=[];predictions=[]
    for seed in p['seeds']:
        refs=np.array(sorted(p['subsets'][str(seed)]['100']));y=x[refs]
        distances=[]
        for start in range(0,500,20):
            q=x[test[start:start+20]]
            distances.append(torch.sqrt((q.square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*q@y.T).clamp_min(0)).cpu().numpy())
        d=np.concatenate(distances);rank=np.argsort(d,axis=1,kind='stable')[:,:5]
        labels=np.array([rows[i]['source_label']=='dog' for i in refs]);guess=labels[rank].mean(1)>.5
        # Direct subtraction independently verifies sample pair distances.
        for qi,ri in [(0,0),(249,99),(499,199)]:
            direct=np.linalg.norm(pixels[test[qi]].astype(np.float64)-pixels[refs[ri]].astype(np.float64))
            assert np.isclose(d[qi,ri],direct,rtol=1e-10,atol=1e-9)
        predictions.append(guess);records.append(dict(seed=seed,accuracy=float((guess==truth).mean()),macro_f1=float(f1_score(truth,guess,average='macro')),p5=float((labels[rank]==truth[:,None]).mean())))
    a=np.array(predictions);np.save(R1/'pixel_predictions.npy',a)
    formal=np.load(R1/'predictions.npz')['resnet18|cosine|raw|clean|100|5']
    errors={}
    for name,label in [('cat',False),('dog',True)]:
        mask=truth==label;counts=(formal[:,mask]!=truth[mask]).sum(1)
        errors[name]=dict(total=int(mask.sum()),seed1001_errors=int(counts[0]),seed1001_rate=float(counts[0]/mask.sum()),ten_seed_mean_errors=float(counts.mean()),ten_seed_mean_rate=float(counts.mean()/mask.sum()))
    result=dict(status='passed',protocol_hash=(OUT/'protocol.sha256').read_text().strip(),scope='Post-hoc descriptive baseline; clean only, shared ten 100/class draws, uniform k=5, stable image-ID ties; no tuning or inferential claim',representation='224x224 RGB locked canvas flattened to 150528 components in [0,1]; raw Euclidean distance; no encoder processor',device=torch.cuda.get_device_name(),precision='float64',accuracy=float((a==truth).mean()),seed_sd=float(np.std([r['accuracy'] for r in records],ddof=1)),macro_f1=float(np.mean([r['macro_f1'] for r in records])),p5=float(np.mean([r['p5'] for r in records])),seeds=records,resnet_class_errors=errors)
    (R1/'supplement.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
