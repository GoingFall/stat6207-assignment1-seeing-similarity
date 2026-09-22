"""Export locked images, distances, metrics and reference-fitted UMAP to site/."""
import json,shutil
import numpy as np
from umap import UMAP
from plotly.offline import get_plotlyjs
from pipelines.evaluate import load_locked
from pipelines.prepare_data import ROOT,OUT
from src.retrieval.distances import pairwise

def main():
    p,rows=load_locked();site=ROOT/'site';(site/'assets').mkdir(parents=True,exist_ok=True);(site/'images').mkdir(exist_ok=True)
    e=json.loads((OUT/'results/evaluation.json').read_text());audit=json.loads((OUT/'results/label_audit.json').read_text())
    refs=np.array(sorted(p['subsets']['1001']['100']));test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);nonclean=[c for c in p['conditions'] if c!='clean']
    for i in np.r_[refs,test]:shutil.copy2(OUT/rows[i]['image'],site/'images'/f"{rows[i]['id']}.png")
    for c in nonclean:shutil.copytree(OUT/c,site/c,dirs_exist_ok=True)
    old=json.loads((site/'assets/data.json').read_text()) if (site/'assets/data.json').exists() else {}
    maps={};ranks={};distances={}
    for model in p['models']:
        f=np.load(OUT/'embeddings'/f'{model}.npy')
        if old.get('protocol_hash')==e['protocol_hash'] and model in old.get('maps',{}):maps[model]=old['maps'][model]
        else:
            mapper=UMAP(n_neighbors=15,min_dist=.1,metric='cosine',random_state=42,transform_seed=42);maps[model]=np.vstack([mapper.fit_transform(f[refs]),mapper.transform(f[test])]).tolist()
        for c in p['conditions']:
            q=f[test] if c=='clean' else f[1500+nonclean.index(c)*500:2000+nonclean.index(c)*500]
            for metric in p['metrics']:
                d=pairwise(q,f[refs],metric);key=f'{model}|{metric}|{c}';ranks[key]=np.argsort(d,axis=1,kind='stable').tolist();distances[key]=d.round(5).tolist()
    payload=dict(protocol=p,protocol_hash=e['protocol_hash'],evaluation=e,audit=audit,reference=[rows[i] for i in refs],queries=[rows[i] for i in test],maps=maps,ranks=ranks,distances=distances)
    (site/'assets/data.json').write_text(json.dumps(payload,separators=(',',':')));(site/'assets/plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    shutil.copytree(OUT/'figures',site/'figures',dirs_exist_ok=True)
    if (ROOT/'report.pdf').exists():shutil.copy2(ROOT/'report.pdf',site/'report.pdf')
    print('Exported current site/')

if __name__=='__main__':main()
