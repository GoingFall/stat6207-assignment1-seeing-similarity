# Version 2.0 results — retrieval, ANN, long-tail learning and open-set monitoring

Version 2.0 extends the frozen Version 1.0 assignment without modifying `releases/1.0/`, `data/`, or the Version 1.0 report. Machine-readable results are under `results_v2/`; immutable local data locks are under `data_v2/manifests/`.

## What the evidence supports

### SOP semantic retrieval

Standard evaluation is leave-one-out retrieval within the official 60,502-image test split. The exact query image is removed by image ID while other images of the same product remain valid positives.

- Exact DINOv2-S/14 Recall@1/10/100: **0.5503 / 0.7171 / 0.8470**.
- Exact mAP@1000: **0.3106**.
- HNSW preserved ANN Recall@10 of **0.9967** and semantic Recall@10 of **0.7150** with a 104.33 MiB index.
- IVF-PQ reduced the index to 3.27 MiB but achieved only **0.4941** ANN Recall@10 and **0.5889** semantic Recall@10. It failed the registered 0.95 development-recall target and is not quality-equivalent to exact search.

Semantic quality and ANN neighbour overlap are distinct claims. SIFT1M results below are not semantic retrieval evidence.

### SIFT1M ANN systems performance

Parameters were selected with queries 0–999 and evaluated once with queries 1,000–9,999. Native Windows `faiss-cpu==1.15.1` is authoritative.

- HNSW held-out Recall@1/10/100: **0.9828 / 0.9813 / 0.9092**; batch-256 throughput **24,734 QPS**; 747.80 MiB.
- IVF-Flat: **0.9812 / 0.9836 / 0.9696**; **1,548 QPS**; 496.04 MiB.
- IVF-PQ: **0.4640 / 0.5631 / 0.6381**; **8,874 QPS**; 23.52 MiB. It did not meet the development Recall@10 target.

The fixed-parameter 10k/50k/120k/1M scale sweep shows HNSW Recall@10 declining from 0.9990 to 0.9813 as scale grows. The 10k IVF-PQ point is deliberately absent because fixed `nlist=1024` cannot satisfy the minimum 39 training vectors per centroid.

The bounded NumPy exact implementation and Faiss Flat agree at Top-1 and Top-10 over the 256-query feasibility set. Their only Top-1 difference from the official IDs is query 219, where two vectors have equal squared L2 distance 47,044; this is tie ordering, not inexact distance computation.

### Balanced and controlled long-tail classification

All classifiers use frozen, unit-normalized 384-D DINOv2-S/14 embeddings. The known-class split has 30/10/10 train/validation/test images for each of 1,000 species.

- Frozen k-NN: **0.5461 accuracy**, **0.5448 macro-F1**.
- Validation-selected linear probe: **0.5761 accuracy**, **0.5685 macro-F1**.

For the controlled 10:1 training distribution, values are mean ± sample standard deviation over the five frozen seeds 6207–6211:

- Cross-entropy: **0.2633 ± 0.0016 accuracy**, **0.0130 ± 0.0027 tail recall**.
- Class-weighted: **0.3616 ± 0.0035 accuracy**, **0.1439 ± 0.0014 tail recall**.
- Balanced sampler: **0.4036 ± 0.0062 accuracy**, **0.3863 ± 0.0073 macro-F1**, **0.3506 ± 0.0078 tail recall**.

Balanced sampling substantially recovers tail performance, with a head-recall trade-off. The registered 50:1 run is blocked because 30 leakage-free candidates per class cannot realize an exact positive-integer 50:1 ratio. It was not rounded, replaced, sampled with replacement, or created using validation/test images.

### Calibration, near-OOD rejection and monitoring

Temperature was fitted only on known-validation. Scaling with **T=0.3619** reduced known-test NLL from **3.1826 to 1.8529** and ECE from **0.4815 to 0.0322** without changing predictions.

Per-image class-novel bird rejection remained weak:

- Best held-out near-OOD AUROC: **0.6359**, from temperature-scaled maximum probability.
- At its development-selected threshold: **0.9545 known coverage**, but only **0.0431 unknown rejection**.
- Nearest-neighbour, agreement-margin and energy AUROC: **0.5878 / 0.6097 / 0.5604**.

The evidence therefore does **not** support a strong per-image open-set detector claim. No far-OOD result is reported because no bird-excluded dataset with adequate publication rights was verified.

Aggregate frozen-embedding monitoring was more sensitive. Thresholds were derived only from repeated disjoint halves of known-validation with one fixed validation-derived RBF bandwidth. Known-test triggered no centroid/PSI/MMD alerts; unknown-test triggered all three. This is a batch-distribution drift result, not a substitute for per-image unknown detection.

## Reproducibility and release boundary

- SOP, SIFT1M and iNaturalist splits/manifests were hashed before inference.
- All 74,300 iNaturalist images were individually SHA-256 checked during encoding.
- The CPU CI contains 14 synthetic/data-free contracts; large data and GPU runs remain manual.
- `/health` and `/v1/status` expose health/artifact booleans without local dataset paths.
- Public packaging excludes source images, embeddings, checkpoints, Faiss indexes and per-query neighbour/score matrices. The iNaturalist image manifest is published only as a path-redacted ID/class/split/hash manifest.
- Optional WSL2 GPU Faiss was not run. It is non-blocking and no native-CPU/WSL-GPU comparison is claimed.

Authoritative details and correction history are in `docs/report/V2-EXECUTION-LOG.md`. Figures are under `results_v2/figures/`.
