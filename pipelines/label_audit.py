"""Independent EfficientNet audit sidecar: never change source labels or selection."""
import json
import numpy as np
import torch
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pipelines.evaluate import load_locked
from pipelines.prepare_data import OUT

def main():
    p,rows=load_locked();test=[r for r in rows if r['split']=='test'];torch.set_num_threads(4)
    weights=EfficientNet_B0_Weights.IMAGENET1K_V1;model=efficientnet_b0(weights=weights).eval().cuda();transform=weights.transforms()
    results=[]
    with torch.inference_mode():
        for start in range(0,len(test),32):
            batch=test[start:start+32];inputs=torch.stack([transform(Image.open(OUT/r['image']).convert('RGB')) for r in batch]).cuda()
            probs=model(inputs).softmax(-1).cpu().numpy()
            for r,prob in zip(batch,probs):
                cat=float(prob[281:286].sum());dog=float(prob[151:269].sum());own,opposite=(cat,dog) if r['source_label']=='cat' else (dog,cat)
                flag='high_confidence_disagreement' if opposite>=.8 and own<=.1 else ('uncertain' if max(cat,dog)<.8 else 'agreement')
                results.append(dict(id=r['id'],source_label=r['source_label'],cat_mass=cat,dog_mass=dog,flag=flag))
    payload=dict(model='EfficientNet-B0 IMAGENET1K_V1',test_manifest_sha256=p['test_manifest_sha256'],protocol_hash=(OUT/'protocol.sha256').read_text(),labels_changed=0,rows=results,note='Independent model predictions are not verified labels; no filtering or relabeling.')
    (OUT/'results/label_audit.json').write_text(json.dumps(payload,indent=2));print('Audit flags:',{flag:sum(r['flag']==flag for r in results) for flag in ['agreement','uncertain','high_confidence_disagreement']},flush=True)

if __name__=='__main__':main()
