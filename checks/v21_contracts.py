"""Regression contracts for review findings, independent of private data."""
import numpy as np
from src.training.metrics import classification_metrics
from src.training.provenance import assert_content_disjoint
from src.benchmarking.ann import sharded_exact_l2


def main():
    y = np.array([0, 1, 2])
    p = np.eye(3)[[0, 0, 2]]
    result = classification_metrics(y, p, {0: 30, 1: 10, 2: 3})
    assert abs(result['groups']['head']['macro_f1'] - 2/3) < 1e-12
    for shard in (3, 7, 20):
        _, ids = sharded_exact_l2(np.zeros((20,2),np.float32), np.zeros((1,2),np.float32),5,1,shard)
        assert ids.tolist() == [[0,1,2,3,4]]
    try:
        assert_content_disjoint([{'image_id':1,'sha256':'same'}, {'image_id':2,'sha256':'same'}])
    except AssertionError: pass
    else: raise AssertionError('Content leakage was accepted')
    from tools.package_v21 import permitted
    for name in ('docs/session/session-transcript.md','data_v21/embeddings/a.npy','results_v21/inat/model.pt','releases/2.0/Assignment1-2.0.zip','../secret.json','results_v21/inat/training_partial.json'):
        assert not permitted(name), name
    assert permitted('data_v21/manifests/lock.json')
    print('V2.1 regression contracts passed')

if __name__ == '__main__': main()
