"""GPU pairwise ResNet Grad-CAM for the first prespecified display query."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.retrieval.encoders import load_encoder
from pipelines.prepare_data import R1,ROOT,OUT

def main():
    torch.set_num_threads(4);data=json.loads((ROOT/'site/v1.0/assets/data.json').read_text());qi=next(i for i,r in enumerate(data['queries']) if r['id']==data['protocol']['display_ids'][0]);query=data['queries'][qi]
    key='resnet18|cosine|clean';rank=data['ranks'][key][qi];candidates=[data['reference'][rank[0]],data['reference'][rank[-1]]]
    model,process,forward=load_encoder('resnet18','cuda');activations=[]
    def capture(module,inputs,output):
        if output.requires_grad:output.retain_grad();activations.append(output)
    hook=model.layer4.register_forward_hook(capture)
    def inp(row):
        with Image.open(OUT/row['image']) as im:return process([im.convert('RGB')])
    with torch.no_grad():refs={r['id']:forward(inp(r)).detach() for r in [query]+candidates}
    def cam(row,reference):
        activations.clear();model.zero_grad(set_to_none=True);inputs=inp(row);embedding=forward(inputs);score=F.cosine_similarity(embedding,reference).sum();score.backward()
        a=activations[-1];heat=(a.grad.mean((2,3),keepdim=True)*a).sum(1,keepdim=True).relu();heat=F.interpolate(heat,size=inputs.shape[-2:],mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy();heat/=heat.max()+1e-12
        rgb=inputs[0].detach().cpu().permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]);return np.clip(rgb,0,1),heat,float(score.detach())
    fig,axes=plt.subplots(2,4,figsize=(10,5.6));records=[]
    for row,candidate in enumerate(candidates):
        for col,(r,reference) in enumerate([(query,refs[candidate['id']]),(candidate,refs[query['id']])]):
            rgb,heat,score=cam(r,reference);axes[row,2*col].imshow(rgb);axes[row,2*col+1].imshow(rgb);axes[row,2*col+1].imshow(heat,cmap='inferno',alpha=.45,vmin=0,vmax=1)
            axes[row,2*col].set_title(r['id'],fontsize=9);axes[row,2*col+1].set_title('Positive similarity evidence',fontsize=8)
        records.append(dict(query=query['id'],candidate=candidate['id'],rank='nearest' if row==0 else 'farthest',cosine_similarity=score))
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('ResNet-18 pairwise cosine Grad-CAM: nearest pair (top), farthest pair (bottom)',fontsize=11);fig.tight_layout();fig.savefig(R1/'figures/gradcam.png',dpi=180);plt.close(fig);hook.remove()
    (R1/'gradcam.json').write_text(json.dumps(dict(device=torch.cuda.get_device_name(),selection='first prespecified display ID; nearest/farthest under ResNet cosine seed1001',pairs=records),indent=2));print('GPU pairwise attribution complete')

if __name__=='__main__':main()
