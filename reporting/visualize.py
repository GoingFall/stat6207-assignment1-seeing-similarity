"""Report evidence from the locked experiment; descriptive plots add no selection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pipelines.prepare_data import ROOT,OUT,R1

def main():
    site=json.loads((ROOT/'site/v1.0/assets/data.json').read_text());data=site['evaluation'];p=site['protocol'];figures=R1/'figures'
    queries=site['queries'];refs=site['reference'];models=p['models'];colors=['#266a9f','#ba6031','#558255']
    qi=next(i for i,r in enumerate(queries) if r['id']==p['display_ids'][0]);q=queries[qi]
    key='resnet18|cosine|clean';rank=site['ranks'][key][qi];ds=site['distances'][key][qi]
    fig,ax=plt.subplots(3,5,figsize=(10,6.8))
    for a in ax.flat:a.axis('off')
    ax[0,2].imshow(Image.open(OUT/q['image']));ax[0,2].set_title(f"Query: {q['id']} ({q['source_label']})",fontsize=10)
    for row,indices in [(1,rank[:5]),(2,rank[-5:][::-1])]:
        for col,ri in enumerate(indices):
            r=refs[ri];ax[row,col].imshow(Image.open(OUT/r['image']));ax[row,col].set_title(f"{r['id']} / {r['source_label']}\nd={ds[ri]:.4f}",fontsize=9)
    fig.suptitle('ResNet-18 / cosine / seed 1001: five nearest (middle), five farthest (bottom)',fontsize=11)
    fig.tight_layout();fig.savefig(figures/'retrieval.png',dpi=180);plt.close(fig)
    dog_id=next(ident for ident in p['display_ids'] if next(r for r in queries if r['id']==ident)['source_label']=='dog')
    qi=next(i for i,r in enumerate(queries) if r['id']==dog_id);q=queries[qi];rank=site['ranks'][key][qi];ds=site['distances'][key][qi]
    fig,axes=plt.subplots(2,6,figsize=(12,4.4))
    for row,indices in enumerate([rank[:5],rank[-5:][::-1]]):
        axes[row,0].imshow(Image.open(OUT/q['image']));axes[row,0].set_title(f"Query {dog_id}\n{'Nearest' if row==0 else 'Farthest'} five",fontsize=9)
        for col,ri in enumerate(indices,1):
            r=refs[ri];axes[row,col].imshow(Image.open(OUT/r['image']));axes[row,col].set_title(f"{r['id']}\nd={ds[ri]:.4f}",fontsize=9)
        for ax in axes[row]:ax.axis('off')
    fig.tight_layout();fig.savefig(figures/'retrieval_dog.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,model in zip(axes,models):
        xy=np.array(site['maps'][model]);allrows=refs+queries
        for label,color in [('cat','#266a9f'),('dog','#ba6031')]:
            for split,marker in [('reference','o'),('test','*')]:
                ids=[i for i,r in enumerate(allrows) if r['source_label']==label and r['split']==split]
                ax.scatter(xy[ids,0],xy[ids,1],s=8 if marker=='o' else 12,c=color,marker=marker,alpha=.65,label=f'{label} {split}')
        ax.set(title=model,xlabel='UMAP 1 (unitless)',ylabel='UMAP 2 (unitless)');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(figures/'umap.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.6))
    for model,color in zip(models,colors):
        points=[next(r for r in data['metrics'] if r['key']==f'{model}|cosine|raw|clean|100|{k}') for k in [1,3,5,7,9]]
        ax.plot([1,3,5,7,9],[r['accuracy']*100 for r in points],marker='o',label=model,color=color)
    ax.set(xlabel='Number of neighbours k (uniform voting)',ylabel='Mean accuracy (%)',xticks=[1,3,5,7,9],title='Descriptive k sensitivity: clean queries, cosine, 100 references/class, ten seeds');ax.legend()
    fig.tight_layout();fig.savefig(figures/'k_sensitivity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    for ax,model in zip(axes,models):
        r=next(r for r in data['metrics'] if r['key']==f'{model}|cosine|raw|clean|100|5');cm=np.mean([s['confusion'] for s in r['seeds']],axis=0)
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=250)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',color='white' if cm[i,j]>125 else 'black')
        ax.set(title=model,xticks=[0,1],yticks=[0,1],xticklabels=['cat','dog'],yticklabels=['cat','dog'],xlabel='Predicted class',ylabel='Source class')
    fig.suptitle('Mean confusion counts across ten seeds (250 queries per source class)',fontsize=11);fig.tight_layout();fig.savefig(figures/'confusion.png',dpi=180);plt.close(fig)
    def get(model,condition='clean',size=100):return next(r for r in data['metrics'] if r['key']==f'{model}|cosine|raw|{condition}|{size}|5')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for model,color in zip(models,colors):
        points=[get(model,size=s) for s in [10,25,50,100]]
        for ax,field in zip(axes,['accuracy','p5']):
            ax.errorbar([10,25,50,100],[r[field] for r in points],yerr=[np.std([s[field] for s in r['seeds']],ddof=1) for r in points],label=model,color=color,marker='o',capsize=3)
            ax.set(xlabel='Reference images per class',ylabel=field+' (fraction)',title='Clean, cosine, k=5; bars = seed SD');ax.legend()
    fig.tight_layout();fig.savefig(figures/'sample_efficiency.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,track in zip(axes,['blur','jpeg','occlusion']):
        for model,color in zip(models,colors):
            points=[get(model,c) for c in ['clean',track+'_mild',track+'_moderate']];y=[r['accuracy'] for r in points];ci=np.array([r['accuracy_ci'] for r in points])
            ax.plot(range(3),y,marker='o',label=model,color=color);ax.fill_between(range(3),ci[:,0],ci[:,1],alpha=.12,color=color)
        ax.set(xticks=range(3),xticklabels=['clean','mild','moderate'],ylabel='Accuracy vs source labels',xlabel='Preset severity',title=track);ax.legend(fontsize=8)
    fig.suptitle('Independent perturbations; cosine, 100/class, k=5; paired-query 95% intervals');fig.tight_layout();fig.savefig(figures/'robustness.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for i,r in enumerate(data['comparisons']):ax.plot(np.array(r['ci'])*100,[i,i],color=colors[0]);ax.scatter(r['difference']*100,i,color=colors[0])
    ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(12),[r['a']+' minus '+r['b'] for r in data['comparisons']],fontsize=8);ax.set(xlabel='Paired accuracy difference (percentage points), 95% query interval',title='Prespecified six-perturbation endpoint, 100/class, k=5');ax.invert_yaxis();fig.tight_layout();fig.savefig(figures/'effects.png',dpi=180);plt.close(fig)
    prediction=np.load(R1/'predictions.npz')['resnet18|cosine|raw|clean|100|5'][0];truth=np.array(data['truth']);fig,axes=plt.subplots(2,5,figsize=(12,5))
    for ax,ident in zip(axes.flat,p['display_ids']):
        i=next(i for i,r in enumerate(queries) if r['id']==ident);r=queries[i];guess='dog' if prediction[i] else 'cat';ax.imshow(Image.open(OUT/r['image']));ax.axis('off');ax.set_title(f"{ident}\nSource {r['source_label']} / {guess}",fontsize=8)
    fig.suptitle('Ten preselected unseen images; ResNet-18 cosine k=5, seed 1001');fig.tight_layout();fig.savefig(figures/'unseen.png',dpi=180);plt.close(fig)
    failures=np.flatnonzero(prediction!=truth)[:4]
    if len(failures):
        fig,axes=plt.subplots(len(failures),6,figsize=(12,2.3*len(failures)),squeeze=False)
        for ri,qi in enumerate(failures):
            q=queries[qi];axes[ri,0].imshow(Image.open(OUT/q['image']));axes[ri,0].set_title(f"{q['id']}\nSource {q['source_label']}",fontsize=8)
            for col,idx in enumerate(site['ranks']['resnet18|cosine|clean'][qi][:5],1):
                r=refs[idx];axes[ri,col].imshow(Image.open(OUT/r['image']));axes[ri,col].set_title(f"#{col}: {r['source_label']}",fontsize=8)
            for ax in axes[ri]:ax.axis('off')
        fig.suptitle('First clean errors by image ID: source-label disagreements, not verified truth');fig.tight_layout();fig.savefig(figures/'failures.png',dpi=180);plt.close(fig)
    print('Generated retrieval, UMAP, k sensitivity and confusion figures from locked results')

if __name__=='__main__':main()
