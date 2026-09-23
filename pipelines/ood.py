"""Twenty original Animals-10 probes; descriptive only, never in binary metrics."""
import json
import numpy as np
import pandas as pd
from pipelines.prepare_data import OUT,ROOT,canvas,digest,R1
from src.retrieval.encoders import encode
from src.retrieval.distances import pairwise

def main():
    p=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'manifest.json').read_text())
    old=pd.read_csv(OUT/'provenance/pilot_manifest.csv');ood=old[old.source.str.contains('animals10')].to_dict('records');assert len(ood)==20
    folder=R1/'ood';folder.mkdir(exist_ok=True);site=ROOT/'site/v1.0';(site/'ood').mkdir(exist_ok=True)
    paths=[]
    for r in ood:
        im=canvas(OUT/'provenance'/r['image']);path=folder/f"{r['id']}.png";im.save(path);im.save(site/'ood'/path.name);paths.append(path)
    refs=np.array(sorted(p['subsets']['1001']['100']));labels=np.array([rows[i]['source_label']=='dog' for i in refs]);results={}
    for model in p['models']:
        cache=folder/f'{model}.npy'
        if cache.exists():q=np.load(cache)
        else:q,_=encode(model,paths,'cuda');np.save(cache,q)
        f=np.load(OUT/'embeddings'/f'{model}.npy');d=pairwise(q,f[refs],'cosine');rank=np.argsort(d,axis=1,kind='stable')[:,:5]
        results[model]=[dict(id=r['id'],source_label=r['label'],prediction='dog' if labels[rank[i]].mean()>.5 else 'cat',nearest=[rows[refs[j]]['id'] for j in rank[i]],distances=d[i,rank[i]].tolist()) for i,r in enumerate(ood)]
    (folder/'results.json').write_text(json.dumps(dict(protocol_hash=(OUT/'protocol.sha256').read_text().strip(),note='Non-cat/dog probes. Forced binary predictions are not OOD detection. No accuracy, threshold or inference claim.',images=[dict(id=r['id'],label=r['label'],sha256=digest(paths[i].read_bytes())) for i,r in enumerate(ood)],models=results),indent=2))
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Animals-10 probes</title><link rel="stylesheet" href="style.css"><main><a href="index.html">Back to formal experiment</a><h1>Beyond cats and dogs</h1><p>Twenty retained Animals-10 probes, shown descriptively. All binary KNN predictions are forced cat/dog outputs; this is not calibrated OOD detection. Cosine, k=5, seed1001, 100 references per class. These images never enter formal metrics.</p>']
    for model,items in results.items():
        html.append(f'<h2>{model}</h2>')
        for r in items:
            html.append(f'<details><summary>{r["id"]}: source {r["source_label"]}; forced prediction {r["prediction"]}</summary><div class="query-row"><img width="160" src="ood/{r["id"]}.png" alt="OOD query"><div class="gallery">')
            for ident,d in zip(r['nearest'],r['distances']):html.append(f'<figure class="image-card"><img src="images/{ident}.png" alt="{ident}"><figcaption>{ident}<br>d={d:.4f}</figcaption></figure>')
            html.append('</div></div></details>')
    html.append('</main></html>');(site/'ood.html').write_text(''.join(html),encoding='utf-8');print('20 Animals-10 supplementary probes exported')

if __name__=='__main__':main()
