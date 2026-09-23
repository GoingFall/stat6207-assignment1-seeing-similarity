"""Fail-closed revision verification, including recomputation from predictions."""
import json
from collections import Counter
import numpy as np
import torch
import faiss
from pipelines.prepare_data import ROOT
from src.training.provenance import sha256, write_json, assert_content_disjoint
from src.training.metrics import classification_metrics, open_set_metrics, coverage_risk, expected_calibration_error
from src.retrieval.open_set import select_threshold
from pipelines.v2_train_inat import aggregate_long_tail
from pipelines.v2_open_set_inat import softmax, temperature_nll
from scipy.optimize import minimize_scalar
from src.paths import resolve


def load(path): return json.loads(resolve(path).read_text(encoding='utf-8'))


def check_artifact(a):
    path = resolve(a['path'])
    assert path.is_file() and sha256(path)==a['sha256'], a['path']
    if 'bytes' in a: assert path.stat().st_size==a['bytes']


def compare(actual, expected, tolerance=1e-7):
    if isinstance(expected,dict):
        for k,v in expected.items(): compare(actual[k],v,tolerance)
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): compare(a,b,tolerance)
    elif isinstance(expected,(int,float)):
        assert abs(actual-expected)<=tolerance,(actual,expected)
    else: assert actual==expected


def main():
    cfg = load('configs/v2.1/experiment.json')
    lock = load('data/v2.1/manifests/lock.json')
    assert sha256(ROOT/'configs/v2.1/experiment.json')==lock['config_sha256']
    encoding = load('results/v2.1/inat/encoding.json')
    assert encoding['status']=='passed'
    check_artifact(encoding['lock'])
    all_rows, split_rows = [], {}
    for split,record in lock['manifests'].items():
        check_artifact(record)
        rows=load(record['path']);split_rows[split]=rows;all_rows.extend(rows)
        a=encoding['splits'][split]['artifacts']
        for value in a.values():check_artifact(value)
        for value in encoding['splits'][split]['source_artifacts'].values():check_artifact(value)
        ids=np.load(resolve(a['image_ids']['path']));labels=np.load(resolve(a['labels']['path']));x=np.load(resolve(a['embeddings']['path']),mmap_mode='r')
        assert ids.tolist()==[r['image_id'] for r in rows]
        assert labels.tolist()==[r['class_id'] for r in rows]
        assert x.shape==(len(rows),384) and np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=2e-6)
        source=encoding['splits'][split]['source_artifacts']
        original_ids=np.load(resolve(source['image_ids']['path']));pos={int(v):i for i,v in enumerate(original_ids)}
        original=np.load(resolve(source['embeddings']['path']),mmap_mode='r')
        assert np.array_equal(x,original[[pos[int(i)] for i in ids]])
    assert_content_disjoint(all_rows)
    assert len({r['image_id'] for r in all_rows})==len(all_rows)
    classes={s:{r['class_id'] for r in rr} for s,rr in split_rows.items()}
    assert classes['train']==classes['validation']==classes['test'] and len(classes['train'])==1000
    assert not (classes['train']&classes['unknown_development'] or classes['train']&classes['unknown_test'] or classes['unknown_test']&classes['unknown_development'])
    for s,n in cfg['split']['known_per_class'].items():assert set(Counter(r['class_id'] for r in split_rows[s]).values())=={n}
    train_ids={r['image_id'] for r in split_rows['train']}
    assert len(lock['long_tail']['runs'])==5 and len(lock['long_tail']['blocked'])==5
    for r in lock['long_tail']['runs']:
        check_artifact(r);sample=load(r['path']);ids=[i for vv in sample['selected_image_ids'].values() for i in vv]
        assert len(ids)==len(set(ids)) and set(ids)<=train_ids
        assert max(sample['counts'].values())/min(sample['counts'].values())==10
        assert all(len(sample['selected_image_ids'][k])==v for k,v in sample['counts'].items())
    training=load('results/v2.1/inat/training.json')
    assert training['status']=='passed' and len(training['long_tail'])==15
    assert training['config_sha256']==lock['config_sha256']
    assert training['embedding_result_sha256']==sha256(ROOT/'results/v2.1/inat/encoding.json')
    assert training['split_lock_sha256']==sha256(ROOT/'data/v2.1/manifests/lock.json')
    assert training['provenance_sha256']==sha256(ROOT/'results/v2.1/provenance.json')
    class_ids=sorted(classes['train']);class_to_index={c:i for i,c in enumerate(class_ids)}
    test_x=np.load(ROOT/'data/v2.1/embeddings/inat_birds/test_embeddings.npy')
    torch.set_num_threads(16)
    for record in [training['balanced_probe']]+training['long_tail']:
        candidates=record['candidates']
        winner=max(candidates,key=lambda c:(c['best_validation_macro_f1'],-c['learning_rate']))
        assert winner['learning_rate']==record['selected_learning_rate']
        assert winner['weight_decay']==record['training']['weight_decay']
        assert len(candidates)==len(cfg['inat_training']['linear_probe']['learning_rates'])*len(cfg['inat_training']['linear_probe']['weight_decays'])
        if 'sample_lock' in record:
            check_artifact(record['model'])
            sampled=load(record['sample_lock']['path'])
            counts={class_to_index[int(c)]:n for c,n in sampled['counts'].items()}
            expected_train_labels=[class_to_index[int(c)] for c,ids in sampled['selected_image_ids'].items() for _ in ids]
        else:
            assert sha256(resolve(record['model_path']))==record['model_sha256']
            counts={i:20 for i in range(1000)}
            expected_train_labels=[class_to_index[r['class_id']] for r in split_rows['train']]
        checkpoint=resolve(record['model']['path'] if 'model' in record else record['model_path'])
        model=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=model['state_dict']
        with torch.inference_mode():
            predicted=torch.nn.functional.linear(torch.from_numpy(test_x),state['weight'],state['bias']).numpy()
        saved=np.load(resolve(record['predictions']['test']['path']))
        assert np.allclose(predicted,saved['logits'],atol=2e-5,rtol=2e-5)
        assert np.array_equal(predicted.argmax(1),saved['predictions'])
        for split,pred in record['predictions'].items():
            check_artifact(pred)
            data=np.load(resolve(pred['path']));p=softmax(data['logits'])
            assert np.array_equal(p.argmax(1),data['predictions'])
            if split in ('validation','test'):
                assert data['labels'].tolist()==[class_to_index[r['class_id']] for r in split_rows[split]]
            else:
                assert data['labels'].tolist()==expected_train_labels
            metrics=classification_metrics(data['labels'],p,counts)
            compare(metrics,pred['metrics'])
            if split=='test':compare(metrics,record['test'])
    check_artifact(training['knn_predictions'])
    d=np.load(resolve(training['knn_predictions']['path']));compare(classification_metrics(d['labels'],d['probabilities'],{i:20 for i in range(1000)}),training['knn'])
    faiss.omp_set_num_threads(16)
    index=faiss.IndexFlatIP(384);index.add(np.load(ROOT/'data/v2.1/embeddings/inat_birds/train_embeddings.npy'))
    _,nn=index.search(test_x,cfg['inat_training']['frozen_knn_k'])
    train_labels=np.array([class_to_index[r['class_id']] for r in split_rows['train']])
    knn=np.zeros_like(d['probabilities'])
    for row,neighbors in enumerate(nn):
        for label,count in Counter(train_labels[neighbors]).items():knn[row,label]=count/len(neighbors)
    assert np.array_equal(knn,d['probabilities'])
    compare(aggregate_long_tail(training['long_tail'],cfg['long_tail']['methods']),training['long_tail_summary'])
    osr=load('results/v2.1/inat/open_set.json');assert osr['status']=='passed'
    assert osr['training_result_sha256']==sha256(ROOT/'results/v2.1/inat/training.json')
    assert osr['scores_sha256']==sha256(resolve(osr['scores_path']))
    scores=np.load(resolve(osr['scores_path']))
    bounds=cfg['inat_training']['calibration']['temperature_bounds']
    opt=minimize_scalar(temperature_nll,bounds=bounds,method='bounded',args=(scores['validation_logits'],scores['validation_labels']),options={'xatol':1e-5})
    assert opt.success and abs(opt.x-osr['calibration']['temperature'])<1e-7
    for name,rec in osr['methods'].items():
        kd,ud,kt,ut=[scores[f'{name}_{s}'] for s in ('known_development','unknown_development','known_test','unknown_test')]
        t=select_threshold(kd,ud,.95);assert t==rec['threshold']
        compare(open_set_metrics(kd,ud,t),{k:v for k,v in rec['development'].items()})
        metrics=open_set_metrics(kt,ut,t)
        pred=scores[f'{name}_test_predictions'];labels=scores['test_labels'];accept=kt>=t
        metrics['known_accuracy_conditional_on_acceptance']=float(np.mean(pred[accept]==labels[accept])) if accept.any() else None
        metrics['coverage_risk']=coverage_risk(labels,pred,kt)
        compare(metrics,rec['test'])
    for side in ('validation','test'):
        ll=scores[f'{side}_logits'];yy=scores[f'{side}_labels']
        for name,t in [('before',1),('after',opt.x)]:
            assert abs(expected_calibration_error(softmax(ll/t),yy)-osr['calibration'][side][f'ece_{name}'])<1e-7
            assert abs(temperature_nll(t,ll,yy)-osr['calibration'][side][f'nll_{name}'])<1e-7
    monitoring=load('results/v2.1/inat/monitoring.json');assert monitoring['status']=='passed'
    check_artifact(monitoring['config_artifact']);check_artifact(monitoring['encoding_artifact'])
    for k,t in monitoring['thresholds'].items():assert t==float(np.quantile([r[k] for r in monitoring['null_records']],.99))
    ref=set(monitoring['reference_rows'])
    from src.monitoring.drift import drift_metrics, estimate_rbf_gamma
    mc=cfg['monitoring'];val_x=np.load(ROOT/'data/v2.1/embeddings/inat_birds/validation_embeddings.npy')
    reference=val_x[monitoring['reference_rows']]
    rng=np.random.default_rng(mc['seed']);rng.permutation(len(val_x))
    projections=rng.standard_normal((mc['random_projections'],384)).astype(np.float32)
    projections/=np.linalg.norm(projections,axis=1,keepdims=True)
    gamma=estimate_rbf_gamma(reference,seed=mc['seed'])
    assert gamma==monitoring['rbf_gamma']
    def measure(xx):return drift_metrics(reference,xx,projections,mc['psi_bins'],mc['rbf_mmd_samples'],mc['seed'],rbf_gamma=gamma)
    for ids,record in zip(monitoring['null_window_rows'],monitoring['null_records']):compare(measure(val_x[ids]),record)
    for ids in monitoring['null_window_rows']:assert len(ids)==1000 and not ref.intersection(ids)
    for split,ev in monitoring['evaluations'].items():
        current=np.load(ROOT/f'data/v2.1/embeddings/inat_birds/{split}_embeddings.npy')
        ids=[i for r in ev['windows'] for i in r['row_ids']];assert len(ids)==len(set(ids))
        for r in ev['windows']:
            compare(measure(current[r['row_ids']]),r['metrics'])
            assert len(r['row_ids'])==1000
            assert r['alerts']=={k:r['metrics'][k]>t for k,t in monitoring['thresholds'].items()}
        compare({k:float(np.mean([r['alerts'][k] for r in ev['windows']])) for k in monitoring['thresholds']},ev['alert_rates'])
    prov=load('results/v2.1/provenance.json')
    exact=load('results/v2.1/checks/exact_audit.json')
    assert exact['status']=='passed' and exact['distance_matrix_matches_flat']
    check_artifact(exact['source_lock']);check_artifact(exact['implementation'])
    for a in prov['frozen_releases'].values():check_artifact(a)
    # Every path listed in the audited layout-migration annex must hash to its recorded value,
    # unless a reviewed post-migration correction starts from exactly those bytes and records the
    # corrected bytes. The annex itself is never rewritten; the correction chain is checked here.
    migration=prov.get('layout_migration',{}).get('files',{})
    corrections=load('results/v2.1/checks/post_migration_corrections.json')['files']
    assert set(corrections)<=set(migration),set(corrections)-set(migration)
    for p,record in corrections.items():
        assert record['migration_post_sha256']==migration[p]['post_migration_sha256'],p
        assert record['corrected_sha256']!=record['migration_post_sha256'],p
    for p,record in migration.items():
        allowed={record['post_migration_sha256']}
        if p in corrections:
            allowed.add(corrections[p]['corrected_sha256'])
        assert sha256(ROOT/p) in allowed,p
    # Training-critical code must match the pre-training source snapshot, or the audited
    # post-layout-migration hash that provenance.json records for that same file.
    for p in ('src/training/metrics.py','pipelines/v2_train_inat.py','pipelines/v21_prepare.py','pipelines/v21_train.py','pipelines/v2_open_set_inat.py'):
        expected={prov['code'][p]}
        if p in migration:
            expected.add(migration[p]['post_migration_sha256'])
        if p in corrections:
            expected.add(corrections[p]['corrected_sha256'])
        assert sha256(ROOT/p) in expected,p
    result={'status':'passed','image_content_unique':len(all_rows),'models_recomputed_from_logits':16,'long_tail_runs':15,'seeds':cfg['long_tail']['seeds'],'checks':['content_and_class_disjointness','source_embedding_identity','locked_config_and_training_code','layout_migration_annex','development_selection','model_prediction_hashes','group_and_overall_metrics','long_tail_aggregate','temperature_refit','ood_thresholds_and_metrics','matched_monitoring_windows','historical_release_hashes','post_migration_corrections']}
    write_json(ROOT/'results/v2.1/verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
