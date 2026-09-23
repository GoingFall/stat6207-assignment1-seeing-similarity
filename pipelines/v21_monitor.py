"""Fixed reference and matched-size window drift demonstration, validation only."""
import json
import numpy as np
from pipelines.prepare_data import ROOT
from src.monitoring.drift import drift_metrics, estimate_rbf_gamma
from src.training.provenance import artifact, write_json


def main():
    target = ROOT/'results_v21/inat/monitoring.json'
    if target.exists(): raise RuntimeError('Completed monitoring exists')
    config_path = ROOT/'configs/v21/experiment.json'
    config = json.loads(config_path.read_text())['monitoring']
    folder = ROOT/'data_v21/embeddings/inat_birds'
    validation = np.load(folder/'validation_embeddings.npy')
    rng = np.random.default_rng(config['seed'])
    order = rng.permutation(len(validation))
    reference_ids = order[:config['reference_images']]
    pool_ids = order[config['reference_images']:]
    reference = validation[reference_ids]
    gamma = estimate_rbf_gamma(reference, seed=config['seed'])
    projections = rng.standard_normal((config['random_projections'],reference.shape[1])).astype(np.float32)
    projections /= np.linalg.norm(projections,axis=1,keepdims=True)
    def measure(current):
        return drift_metrics(reference,current,projections,config['psi_bins'],config['rbf_mmd_samples'],config['seed'],rbf_gamma=gamma)
    null, windows = [], []
    for _ in range(config['calibration_partitions']):
        ids = rng.choice(pool_ids,config['window_size'],replace=False)
        windows.append(ids.tolist())
        null.append(measure(validation[ids]))
    keys = ('centroid_cosine_shift','projection_psi','rbf_mmd')
    thresholds = {k:float(np.quantile([r[k] for r in null],config['alert_quantile'])) for k in keys}
    evaluations = {}
    for split in config['evaluation_splits']:
        x = np.load(folder/f'{split}_embeddings.npy')
        perm = np.random.default_rng(config['seed']).permutation(len(x))
        records = []
        for begin in range(0,len(x)-config['window_size']+1,config['window_size']):
            ids = perm[begin:begin+config['window_size']]
            metrics = measure(x[ids])
            records.append({'row_ids':ids.tolist(),'metrics':metrics,'alerts':{k:metrics[k]>thresholds[k] for k in keys}})
        evaluations[split] = {'windows':records,'unused_images':len(x)%config['window_size'],'alert_rates':{k:float(np.mean([r['alerts'][k] for r in records])) for k in keys}}
    write_json(target, {'status':'passed','config':config,'config_artifact':artifact(config_path,ROOT),'encoding_artifact':artifact(ROOT/'results_v21/inat/encoding.json',ROOT),'reference_rows':reference_ids.tolist(),'null_window_rows':windows,'null_records':null,'rbf_gamma':gamma,'thresholds':thresholds,'evaluations':evaluations,'interpretation':'Repeated validation windows overlap; empirical thresholds and held-out batch demonstration, not a certified independent 1% false-alarm guarantee'})
    print(json.dumps({'thresholds':thresholds,'alert_rates':{k:v['alert_rates'] for k,v in evaluations.items()}},indent=2))


if __name__ == '__main__': main()
