# Version 2.0 execution log

This log records completed gates and protocol corrections. Machine-readable measurements are stored under `results_v2/`; immutable dataset manifests are stored under `data_v2/manifests/`.

## M0 — environment and data gates

- Native Windows is the authoritative platform; `faiss-cpu==1.15.1` and `psutil==7.2.2` are installed.
- RTX 5070 Laptop GPU (`sm_120`) passed the sustained DINOv2-S/14 CUDA gate: outputs stayed on CUDA, 4,700 forwards over 10.09 seconds, 465.68 images/s, and 88.82% mean utilization among active `nvidia-smi` samples.
- Native CPU Faiss Flat, HNSW, IVF-Flat and IVF-PQ smoke tests passed. A first smoke-test defect that added vectors twice to Flat was detected because Recall@10 was 0.5; the harness was corrected and exact Recall@1/@10 became 1.0.
- SOP Kaggle mirror is a TensorFlow Datasets export with 16 train and 16 test TFRecord shards. Its official product-disjoint split is recoverable.
- Official SIFT1M source and CC0 statement were verified at `http://corpus-texmex.irisa.fr/`; the official archive is distributed over FTP.

## M1 — SOP preparation

- The plan’s earlier phrase “query and gallery separate by product ID” was corrected before inference because it would remove every relevant item. Standard SOP evaluation uses the official product-disjoint train/test boundary and leave-one-out retrieval within test.
- Frozen SOP manifests contain 59,551 train images across 11,318 products and 60,502 test images across 11,316 products. Product overlap is zero; every product has at least two images.
- Standard semantic retrieval uses only the 60,502-image test split. A 120,053-image combined index, if measured, is systems-capacity evidence only and is not a standard semantic retrieval result.
- The SOP mirror is labelled only “Data files © Original Authors”; raw SOP images are treated as local-only and are excluded from public Version 2.0 packaging.

## M3 data-rights gate

- The iNaturalist Birds mirror says CC0, but official iNaturalist 2021 terms restrict use to non-commercial research/education and prohibit redistributing images. The stricter official terms govern release behavior.
- Animals-90 is not accepted as far-OOD: it includes birds and has unclear Google Images redistribution rights. It is omitted unless both taxonomy and rights are resolved.
- Frozen unknown-development/test species are explicitly near-OOD class-novel birds from the same iNaturalist Birds subset. No far-OOD metric will be reported in Version 2.0 unless a bird-excluded source with adequate publication rights is independently verified.
- Metadata confirms exactly 50 images per bird species. A literal 50:1 downsampled training ratio is incompatible with an independent 30/10/10 train/validation/test allocation. The preregistered 10:1 and 50:1 ratios remain unchanged; the 50:1 run is explicitly blocked rather than being replaced, rounded, or produced with leakage.
- Before any iNaturalist embedding or training run, the frozen-feature protocol was completed in `configs/v2/experiment.json`: KNN uses k=10; linear probes select learning rate from 1e-4/3e-4/1e-3 by validation macro-F1 with fixed weight decay, batch, epoch and patience; calibration uses 15-bin ECE; rejection thresholds target 95% known-development coverage and are selected only with known-validation plus unknown-development.
- Five exact 10:1 sampled-ID/count locks were written for seeds 6207–6211. Five corresponding 50:1 blocker records were written because 30 available training candidates per class cannot realize an exact positive-integer 50:1 ratio.
- Monitoring was preregistered before embedding generation: known-validation is the only reference/threshold source; known-test and unknown-test are evaluated using centroid cosine shift, random-projection PSI and unbiased RBF-MMD. Test sets cannot change alert thresholds.
- The restricted iNaturalist Birds mirror was acquired locally after M2 completion. All 74,300 frozen image IDs were uniquely resolved and hashed before inference; total image bytes are 5,507,911,024 and the image manifest SHA-256 is `353b26827bbe81a53a408fd1b7d809fc3fbf6ce06c28b82aa5a93766a93f8d69`. Raw images remain excluded from publication.
- DINOv2-S/14 encoded all 74,300 locked images as unit-normalized 384-D float32 CLS embeddings after rechecking each source-image SHA-256. The five splits completed in approximately 444 seconds total; per-split throughput ranged from 150.68 to 177.72 images/s, peak allocated GPU memory was 490,125,824 bytes, and peak sampled GPU memory was 825 MiB.
- Validation-only monitoring thresholds were centroid cosine shift 0.002207, random-projection PSI 0.004128 and fixed-bandwidth RBF-MMD 0.000127. Known-test remained below all three thresholds (no alerts); class-novel unknown-test exceeded all three (three alerts). The RBF bandwidth was estimated once from known-validation and reused for every null and evaluation comparison.
- On the balanced known-class test set, frozen-feature k-NN achieved 0.5461 accuracy / 0.5448 macro-F1 / 0.1307 ECE. The validation-selected balanced linear probe improved accuracy to 0.5761 and macro-F1 to 0.5685, but its uncalibrated ECE was 0.4815; calibration is therefore reported separately rather than implied by classification accuracy.
- Across five frozen 10:1 sampling seeds, ordinary cross-entropy achieved 0.2633 ± 0.0016 accuracy and only 0.0130 ± 0.0027 tail recall. Class weighting improved these to 0.3616 ± 0.0035 and 0.1439 ± 0.0014. Balanced sampling performed best overall at 0.4036 ± 0.0062 accuracy, 0.3863 ± 0.0073 macro-F1 and 0.3506 ± 0.0078 tail recall, while reducing head recall to 0.4471 ± 0.0120. Values are mean ± sample standard deviation across all five preregistered seeds.
- The 50:1 experiment remains blocked, not approximated: with 30 leakage-free training candidates per class, an exact positive-integer 50:1 schedule is infeasible.
- Temperature scaling fitted only on known-validation selected T=0.3619. On known-test it reduced NLL from 3.1826 to 1.8529 and ECE from 0.4815 to 0.0322 without changing class predictions.
- Near-OOD class-novel bird rejection was weak despite the clear aggregate drift signal. The best held-out AUROC was 0.6359 for temperature-scaled maximum probability; at its development-selected threshold it retained 0.9545 known-test coverage but rejected only 0.0431 unknown-test images. Nearest-neighbour similarity, agreement margin and energy AUROC were 0.5878, 0.6097 and 0.5604 respectively. These results do not support a strong per-image open-set detector claim, and no far-OOD claim is made.

## Completed data and encoding work

- SOP DINOv2 encoding completed for all 59,551 train and 60,502 test images. Train/test throughput was 161.97/171.68 images/s; embeddings are 384-D unit-normalized float32 with verified SHA-256. Full-pipeline active-sample GPU utilization (including decode/I/O gaps) was 52.81%/54.82%, distinct from the sustained-forward gate.
- Standard 60,502-image SOP exact/ANN evaluation completed and passed its corrected audit.
- Official SIFT1M download completed. The 168,280,445-byte archive matches official MD5 `b23d1b3b2ee8469d819b61ca900ef0ed` and SHA-256 `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`. Extracted shapes are 1,000,000×128 base, 10,000×128 query, 100,000×128 learn and 10,000×100 ground truth; all files have frozen SHA-256 hashes.
- The native Windows CPU SIFT1M benchmark, iNaturalist image acquisition, frozen embedding generation, training, calibration, near-OOD evaluation and monitoring all completed.

## M2 — SIFT1M native CPU benchmark

- The bounded exact path processed 256 queries in 1.876 seconds (136.47 QPS) using 256-query blocks and 100,000-vector database shards; measured peak RSS was 1.69 GiB.
- Its official-ID overlap was Recall@1/10/100 = 0.9961/0.9996/1.0000. An independent audit showed the sharded implementation and Faiss Flat agree at Top-1 and Top-10. The sole official Top-1 difference, query 219, is an equal-distance tie: IDs 264459 and 200181 both have squared L2 distance 47044. Therefore this is tie ordering, not an inexact computation.
- Development-only selection chose HNSW (`M=32`, `efSearch=64`), IVF-Flat (`nlist=256`, `nprobe=16`) and IVF-PQ (`nlist=1024`, `nprobe=64`, 16×8-bit codes). Held-out Recall@1/10/100 was 0.9828/0.9813/0.9092, 0.9812/0.9836/0.9696 and 0.4640/0.5631/0.6381 respectively.
- HNSW achieved 24,734 batch-256 QPS at 0.046 ms normalized p95/query with a 747.80 MiB index. IVF-Flat achieved 1,548 QPS at 0.703 ms and 496.04 MiB. IVF-PQ achieved 8,874 QPS at 0.131 ms and 23.52 MiB, but did not meet the 0.95 development Recall@10 target and was selected only by the preregistered fallback.
- These are SIFT1M ANN systems results only; no semantic retrieval claim is attached to them.
- The preregistered 10k/50k/120k/1M scale sweep completed using the 1M-selected parameters without per-scale retuning. HNSW Recall@10 was 0.9990/0.9962/0.9934/0.9813 while batch QPS declined from 212,000 to 24,734. IVF-Flat Recall@10 was 0.9234/0.9578/0.9713/0.9836 while QPS declined from 208,613 to 1,548. IVF-PQ Recall@10 was 0.6328/0.6170/0.5631 at 50k/120k/1M; the 10k run was skipped because fixed `nlist=1024` cannot satisfy 39 training vectors per centroid.
- Exact Flat throughput at 10k/50k/120k was 49,309/11,270/3,010 QPS. The 1M exact feasibility result uses the separately preregistered bounded 256-query sharded path and is not replaced by a full 9,000-query Flat timing.

## M5 — CPU CI scaffold

- `.github/workflows/v2-cpu-contracts.yml` compiles Version 2.0 modules and runs small synthetic contracts on Windows/Python 3.11.
- Contracts cover sharded exact L2 equivalence to Faiss Flat, HNSW recall, exact image-ID self exclusion, semantic metrics, product-disjoint splits, development-only rejection threshold logic and SHA-256 manifest verification.
- A dependency-free service exposes `/health` and `/v1/status`; CI starts it on an ephemeral localhost port and verifies 200/404 behavior without loading embeddings or exposing local dataset paths.
- GPU and full-data benchmarks remain manual; CI does not claim hardware-performance reproducibility.
- The optional WSL2 GPU Faiss smoke test was not run. It was explicitly non-blocking, and no cross-platform CPU/GPU comparison is claimed.
- Final authoritative verification passed SOP/SIFT/iNaturalist locks, embeddings, retrieval, ANN scales, five-seed long-tail training, open-set outputs and monitoring. CPU CI passes 14 data-free contracts.

## SOP benchmark audit

- The first standard run produced valid exhaustive semantic metrics: Recall@1/10/100 = 0.5503/0.7171/0.8470 and mAP@1000 = 0.3106.
- Its IVF candidates emitted Faiss under-training warnings because only 10,000 vectors were supplied for 1,024/4,096 centroids. That run is retained as `results_v2/sop/retrieval-superseded-undertrained-ivf.json` but its IVF selections are not accepted.
- The rerun trains IVF quantizers on the complete unlabeled gallery and admits only `nlist` values satisfying Faiss's minimum 39 training vectors per centroid. Held-out query results remain excluded from parameter selection.
- The same pre-run feasibility rule excludes `nlist=4096` from SIFT1M IVF selection because the official 100,000-vector learning set is below Faiss's recommended 159,744 vectors. The preregistered grid remains visible in configuration; the exclusion and reason are written into the result.
- Before M2 execution, the exact feasibility query count was corrected from 100 to 256 so the measured path actually exercises the preregistered 256-query block against 100,000-vector database shards; no SIFT1M benchmark result existed when this correction was made.
- The corrected SOP run passed. Exhaustive cosine-equivalent search measured 2,480 QPS at batch 256 with 0.585 ms normalized per-query p95. Held-out ANN Recall@1/10/100 was 0.9994/0.9967/0.8874 for HNSW (`M=32`, `efSearch=32`), 0.9831/0.9677/0.9114 for IVF-Flat (`nlist=1024`, `nprobe=16`), and 0.4380/0.4941/0.5844 for IVF-PQ (`nlist=1024`, `nprobe=64`, 16×8-bit codes).
- HNSW, IVF-Flat and IVF-PQ serialized sizes were 104.33 MiB, 90.60 MiB and 3.27 MiB. IVF-PQ did not meet the 0.95 development Recall@10 target; per the preregistered fallback it was selected by highest development recall, not presented as quality-equivalent to exact search.
- Corresponding held-out semantic Recall@1/10/100 was 0.5498/0.7150/0.8380 for HNSW, 0.5441/0.7032/0.8207 for IVF-Flat, and 0.3968/0.5889/0.7654 for IVF-PQ. These semantic values remain separate from ANN recall.
- `reporting.v2_results` writes semantic quality and ANN systems trade-offs to separate figures plus `results_v2/summary.json`; it never plots SIFT1M as semantic retrieval quality.
