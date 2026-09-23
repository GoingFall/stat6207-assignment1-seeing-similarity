"""Verify corrected exact code on SIFT without replacing historical results."""
import json
import faiss
import numpy as np
from pipelines.prepare_data import ROOT
from src.benchmarking.vector_io import read_fvecs, read_ivecs
from src.benchmarking.ann import sharded_exact_l2, recall_metrics
from src.training.provenance import artifact, write_json


def main():
    target=ROOT/'results_v21/checks/exact_audit.json'
    if target.exists():raise RuntimeError('Exact revision audit already exists')
    lock=json.loads((ROOT/'data_v2/manifests/sift1m_lock.json').read_text())
    base=read_fvecs(ROOT/lock['files']['base']['path'])
    q=read_fvecs(ROOT/lock['files']['query']['path'])[:256]
    official=read_ivecs(ROOT/lock['files']['groundtruth']['path'])[:256]
    faiss.omp_set_num_threads(16)
    d,ids=sharded_exact_l2(base,q,100,256,100000)
    flat=faiss.IndexFlatL2(128);flat.add(base);fd,fi=flat.search(q,100)
    assert np.array_equal(d,fd)
    for row in range(len(q)):
        for value in np.unique(d[row]):
            tied=ids[row][d[row]==value]
            assert np.array_equal(tied,np.sort(tied))
    result={'status':'passed','queries':256,'distance_matrix_matches_flat':True,'flat_id_overlap':recall_metrics(fi,ids),'official_id_overlap':recall_metrics(official,ids),'source_lock':artifact(ROOT/'data_v2/manifests/sift1m_lock.json',ROOT),'implementation':artifact(ROOT/'src/benchmarking/ann.py',ROOT),'scope':'Correctness audit only; no revised ANN throughput measurements'}
    write_json(target,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
