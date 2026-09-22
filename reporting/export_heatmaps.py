"""Exact GAP-layer ResNet pairwise Grad-CAM, exported compactly for static browsing.

For pooled feature f=mean(A), d score/d A = (d score/d f)/49.
This equals autograd Grad-CAM at layer4; L1/L2 target negative distance,
cosine targets similarity. Stores 7x7 positive maps for ten shown neighbours.
"""
import json
import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps
from src.retrieval.encoders import load_encoder
from pipelines.prepare_data import ROOT,OUT,digest

def main():
    torch.set_num_threads(4);site=ROOT/'site';d=json.loads((site/'assets/data.json').read_text());conditions=list(d['protocol']['conditions']);refs=d['reference'];queries=d['queries'];folder=site/'heatmaps';folder.mkdir(exist_ok=True)
    model,process,forward=load_encoder('resnet18','cuda');captured=[]
    hook=model.layer4.register_forward_hook(lambda m,i,o:captured.append(o.detach()))
    paths=[OUT/r['image'] for r in refs]+[OUT/r['image'] if c=='clean' else OUT/c/f"{r['id']}.png" for c in conditions for r in queries]
    names=[f"reference/{r['id']}" for r in refs]+[f"{c}/{r['id']}" for c in conditions for r in queries]
    acts=[];features=[]
    with torch.no_grad():
        for start in range(0,len(paths),32):
            images=[]
            for path in paths[start:start+32]:
                with Image.open(path) as im:images.append(im.convert('RGB'))
            inputs=process(images);captured.clear();f=forward(inputs);acts.append(captured[-1]);features.append(f)
            rgb=(inputs.cpu().permute(0,2,3,1).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]))
            for name,pixels in zip(names[start:start+32],rgb):
                path=folder/'views'/f'{name}.jpg';path.parent.mkdir(exist_ok=True,parents=True);Image.fromarray(np.uint8(np.clip(pixels,0,1)*255)).save(path,quality=90)
    hook.remove();a=torch.cat(acts);f=torch.cat(features);assert torch.allclose(a.mean((2,3)),f,atol=1e-5)
    records=[];max_error=0
    for ci,c in enumerate(conditions):
        qf=f[200+ci*500:200+(ci+1)*500];qa=a[200+ci*500:200+(ci+1)*500]
        for metric in ['l1','l2','cosine']:
            rank=np.array(d['ranks'][f'resnet18|{metric}|{c}']);ids=np.concatenate([rank[:,:5],rank[:,-5:][:,::-1]],axis=1);idx=torch.as_tensor(ids,device='cuda');x=qf[:,None,:].expand(-1,10,-1);y=f[idx]
            if metric=='cosine':
                nx=x.norm(dim=-1,keepdim=True);ny=y.norm(dim=-1,keepdim=True);cos=(x*y).sum(-1,keepdim=True)/(nx*ny);gx=y/(nx*ny)-cos*x/nx.square();gy=x/(nx*ny)-cos*y/ny.square()
            elif metric=='l1':gx=-(x-y).sign();gy=-gx
            else:gx=-(x-y)/(x-y).norm(dim=-1,keepdim=True).clamp_min(1e-12);gy=-gx
            qm=torch.einsum('qnd,qdhw->qnhw',gx,qa).relu()/49;rm=torch.einsum('qnd,qndhw->qnhw',gy,a[idx]).relu()/49
            maps=torch.stack([qm,rm],dim=2);maps=maps/maps.amax(dim=(-1,-2),keepdim=True).clamp_min(1e-12);raw=(maps*255).round().byte().cpu().numpy().tobytes();path=folder/f'{c}-{metric}.bin';path.write_bytes(raw)
            # Independent autograd verification for both image sides, all 21 conditions/metrics.
            ax=qa[0:1].detach().requires_grad_();ay=a[idx[0,0]:idx[0,0]+1].detach().requires_grad_();fx=ax.mean((2,3));fy=ay.mean((2,3))
            score=torch.nn.functional.cosine_similarity(fx,fy).sum() if metric=='cosine' else (-(fx-fy).abs().sum() if metric=='l1' else -(fx-fy).norm())
            gg=torch.autograd.grad(score,(ax,ay));expected=torch.stack([(gg[0].mean((2,3),keepdim=True)*ax).sum(1).relu()[0],(gg[1].mean((2,3),keepdim=True)*ay).sum(1).relu()[0]])
            expected=expected/expected.amax(dim=(-1,-2),keepdim=True).clamp_min(1e-12);err=float((expected-maps[0,0]).abs().max().detach());max_error=max(max_error,err);assert err<2e-5
            records.append(dict(condition=c,metric=metric,bytes=len(raw),sha256=digest(raw)))
            print('Exported',c,metric,flush=True)
    meta=dict(model='resnet18',layer='layer4',shape=[500,10,2,7,7],order='query, nearest five then farthest five, query/candidate, y, x',conditions=conditions,metrics=['l1','l2','cosine'],protocol_hash=d['protocol_hash'],palette=np.uint8(colormaps['inferno'](np.linspace(0,1,256))[:,:3]*255).tolist(),files=records,note='Positive evidence for pairwise similarity (cosine) or negative distance (L1/L2); per-map normalized. Not a KNN voting derivative.')
    (folder/'index.json').write_text(json.dumps(meta,separators=(',',':')));(OUT/'results/heatmap_check.json').write_text(json.dumps(dict(status='passed',maps=500*10*2*21,autograd_pairs=21,max_normalized_error=max_error,device=torch.cuda.get_device_name(),protocol_hash=d['protocol_hash']),indent=2));print('All heatmap exports verified',flush=True)

if __name__=='__main__':main()
