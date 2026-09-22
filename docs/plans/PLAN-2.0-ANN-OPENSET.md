# Version 2.0 plan — training, ANN retrieval and open-set evaluation

**Status:** approved and in execution; see `docs/report/V2-EXECUTION-LOG.md` for completed gates, protocol corrections and active runs.  
**Baseline:** Version 1.0 remains frozen under `releases/1.0/`; this plan must not modify its report, protocol, data or archive.  
**Primary platform:** Windows 11 native, CPU Faiss.  
**Optional platform:** WSL2, one non-blocking GPU-Faiss smoke test only.

Active implementation follows the repository package layout: reusable code in `src/`, experiment entry points in `pipelines/`, artifacts in `reporting/`, verification in `checks/`, and release/deployment commands in `tools/`. All commands run from the repository root as `python -m package.module`.

## 0. Scope and decisions before implementation

The objective is to add credible evidence for training/tuning/evaluation and production-style retrieval without taking on GLDv2 (85 GB index) or iNaturalist 2018 (120 GB images). The proposed data budget is approximately 10 GB before derived embeddings and indices.

Two claims will remain separate:

1. **Retrieval quality:** whether returned images are relevant under a labelled image-retrieval task.
2. **ANN systems performance:** how closely an index approximates exact neighbours and the latency/QPS/memory trade-off.

No repeated copies of the existing images will be described as a million independent image corpus. Standard ANN vectors will be used for the million-scale systems benchmark.

## 1. Data plan

### 1.1 Stanford Online Products — real retrieval quality

- Candidate source: Kaggle `ryanholbrook/stanford-online-products`; verify its contents against the official Stanford Online Products release before use.
- Approximate size: 120k images, 22.6k product-instance classes, approximately 3 GB.
- **Preferred split:** the official train/test split by product ID, if the Kaggle mirror contains it.
- **Fallback split:** construct exactly once before any retrieval experiment, using product IDs rather than image-level random splitting. Save the sorted train and test product-ID lists and image manifest, then compute SHA-256 hashes. The split is immutable after the first embedding or index is produced.
- The official train and test splits are separate by product ID. Standard retrieval evaluation is leave-one-out within the official test split: every test image is a query and all other test images form its gallery, so other images with the same product ID are positives.
- For every query, exclude only the query image by its exact image ID—not its product ID—and retain other same-product images as valid positives. This corrects the earlier contradictory wording that would have removed every positive from the gallery.
- The verification script must assert:
  - each query image is excluded from its own returned candidates while other images from its product remain eligible;
  - every query has at least one other same-product gallery image;
  - the returned top-1 image ID is not the query's own image ID;
  - query/product split has no product-ID overlap;
  - split and image manifests match their recorded SHA-256 values.

### 1.2 Million-vector ANN benchmark

- Use a standard public ANN benchmark such as **SIFT1M** for the primary 1M × 128-D system test.
- This is a vector/index benchmark, not an image-semantic result. Report it separately from SOP.
- Include exact ground truth supplied by the benchmark. “Exact” means exhaustive comparison against every database vector using the selected distance, retaining the true top-k with deterministic tie handling.
- If a standard benchmark download or licensing check is not convenient, use a clearly labelled synthetic/vector stress test only as a capacity test, never as evidence of image retrieval quality.

### 1.3 Multi-class and open-set image evaluation

- Primary candidate: Kaggle `sharansmenon/inat2021birds`, about 74k images and 1,486 bird species. Labels and metadata must be checked against the official source. Although the mirror page says CC0, the official iNaturalist 2021 terms restrict the dataset to non-commercial research/education and prohibit image redistribution; those stricter terms govern this project. A public portfolio release may include code, hashes and aggregate results, but not dataset images.
- It is a train-mini-style dataset with approximately 50 images per species, so it is **not** a natural long-tail benchmark.
- `Animals-90` is **not** accepted as an automatically valid far-OOD set because it includes bird categories. Its Kaggle license is also only “Other”, and its description says images were collected from Google Images without per-image redistribution terms. It will not be used unless both a bird-excluded class list and adequate image rights can be verified; the current default is to omit it.
- All OOD class lists must be fixed before threshold tuning and hashed.

## 2. Hardware and platform gates

Current machine inventory:

- AMD Ryzen 9 7945HX, 16 cores / 32 threads;
- 32 GB RAM;
- RTX 5070 Laptop GPU, approximately 8 GB VRAM, compute capability 12.0 (`sm_120`);
- Windows native CPU path is the default;
- WSL2 GPU path is optional and non-blocking.

### 2.1 GPU sanity gate before any large encoding

Before starting the iNaturalist or SOP experiment, run a fixed 10-minute or 100-image DINOv2-S/14 encoding benchmark:

- record PyTorch, CUDA, driver, GPU name and compute capability;
- record images/sec and wall time;
- sample `nvidia-smi` utilization and memory at a fixed interval;
- require sustained high GPU utilization (target >80% during model forward, allowing I/O gaps);
- record whether tensors and outputs are on CUDA;
- fail the gate if the run silently falls back to CPU.

The existing successful CUDA detection is not sufficient: Blackwell support has had version-dependent compatibility issues. If the gate fails, test a supported PyTorch build (stable or nightly as appropriate), then rerun the same benchmark. Do not silently continue and report CPU execution as GPU performance.

### 2.2 Memory and timing instrumentation

Every encoding and index benchmark must record:

- wall time and throughput;
- CPU RSS using `psutil.Process().memory_info().rss`;
- peak GPU allocation using `torch.cuda.max_memory_allocated()` and, where available, peak reserved memory;
- index build time, serialized index size and process RSS before/after build;
- search p50/p95/p99 latency and QPS under fixed warm-up, batch and concurrency conditions.

## 3. Native Windows Faiss plan

Faiss GPU has no official native Windows support. Therefore:

1. **Default:** use `faiss-cpu` on native Windows for every primary index experiment.
2. **Optional:** install/use Faiss GPU under WSL2 and run one smoke test after the CPU results are complete. If installation, CUDA linkage or correctness fails, record the failure and abandon the optional path. It must not block or change the main result.
3. Do not compare native CPU and WSL2 GPU numbers as if they were a controlled hardware benchmark unless environment and measurement conditions are explicitly documented.

### 3.1 Indexes

For each dataset/scale, compare:

- `IndexFlatL2` or exact cosine-equivalent normalized search;
- HNSW, starting with fixed `M=32`, `efConstruction=200`, and a registered `efSearch` grid;
- IVF-Flat, with fixed `nlist` and `nprobe` grids;
- IVF-PQ, with registered code size and search parameters.

No parameter may be selected after inspecting the final test result. Use a development query set for selecting defaults and a held-out benchmark query set for final reporting.

### 3.2 Exact-search feasibility test

On 1M × 128-D float32 vectors, exhaustive exact search requires roughly 512 MB for the database vectors before temporary buffers. It is computationally expensive but CPU-feasible as a batched reference on 16 cores; measure it rather than infer it. Start with:

- query blocks of 256;
- database shards of 100,000;
- deterministic top-k merge across shards;
- pre-registered values in the experiment configuration.

For the current 8 GB GPU, do not allocate a full query-by-database distance matrix. A nominal 256 × 1,000,000 float32 matrix is approximately 0.95 GiB, before model/runtime workspace; a 1024 block is approximately 3.8 GiB. Keep the CPU-sharded path as the authoritative exact reference unless a measured GPU path is independently verified.

### 3.3 Required ANN metrics

- ANN Recall@1, @10 and @100 against exact neighbours;
- SOP retrieval Recall@k/mAP against same-product ground truth;
- p50/p95/p99 search-only latency;
- single-query and fixed-batch QPS;
- index build time, serialized size and peak RSS;
- recall–latency–memory curves.

ANN recall and semantic retrieval quality must appear in separate tables and plots.

## 4. Controlled long-tail protocol

Because iNaturalist Birds is approximately balanced at 50 images/species, construct a controlled long-tail training distribution. Do not claim it represents natural iNaturalist imbalance.

- Preserve a fixed balanced validation/test image pool where possible.
- Configure exactly two imbalance ratios: **10:1** and **50:1** (`max_train_count:min_train_count`), as preregistered. Metadata inspection confirmed exactly 50 images/species. Because an independent validation/test allocation leaves fewer than 50 training images, the 50:1 run is blocked until a leakage-free protocol or additional licensed images are available; it must not be silently replaced by 40:1, rounded, or created with validation/test leakage.
- Write the sampling rule, maximum/minimum counts, class ordering and random seed into a versioned YAML/JSON configuration.
- Use at least five fixed seeds per ratio; report mean and standard deviation.
- Save the exact class-count vector and selected image IDs with SHA-256.
- Compare ordinary cross-entropy, class-weighted cross-entropy and balanced sampling using the same split and augmentation budget.
- Report head/medium/tail macro-F1, per-group recall, balanced accuracy and calibration; do not rely on one aggregate accuracy.
- No imbalance ratio or loss choice may be selected using the final test set.

## 5. Open-set and calibrated rejection protocol

Use class-level separation:

- known train/validation/test species;
- unknown-development species for threshold selection;
- unknown-test species held out from threshold selection;
- near-OOD species selected by taxonomic proximity where metadata supports it;
- far-OOD only after exact class-level exclusion is verified.

Methods, from simplest to more advanced:

- nearest-neighbour distance threshold;
- neighbour agreement and class-margin threshold;
- linear-probe maximum probability with temperature scaling;
- energy score.

Thresholds are selected only on known validation plus unknown-development data. Final reporting includes AUROC, AUPR-OOD, FPR@95TPR, known-class coverage–risk and accuracy conditional on acceptance. Calibration (ECE/reliability) and rejection are reported as separate concepts.

## 6. Training experiment connection

After the data and split gates are frozen:

- retain frozen-feature KNN as the baseline;
- train a linear probe with a validation-selected regularization grid;
- optionally test LoRA on one encoder only if the GPU sanity gate passes;
- use training/validation/test separation and record learning curves;
- compare performance, calibration, compute, memory and overfitting gap;
- keep all final test results separate from tuning.

The expected interview answer is empirical: frozen features are a low-variance small-data baseline; linear probing tests separability without changing the encoder; LoRA/finetuning may adapt the representation but can overfit the limited labelled set. No claim is made before the experiment.

## 7. Milestones and stop conditions

### M0 — gates and data inspection

Verify Kaggle contents, licenses, split files, GPU benchmark, native Faiss installation and available storage. Stop if SOP does not contain a recoverable official split or if a custom split cannot be frozen before inference.

### M1 — SOP + native CPU Faiss

Run split hashing, exact search, HNSW and IVF-PQ on a small scale, then on the official 60,502-image test gallery. The full 120,053-image train+test index may be measured only as a separately labelled systems-capacity stress test because combining product-disjoint splits changes the standard semantic task. Stop if resource measurements threaten the machine or if retrieval labels are unusable.

### M2 — 1M × 128-D ANN

Run SIFT1M exact, HNSW, IVF-Flat and IVF-PQ with pre-registered grids. Stop before adding WSL2 until CPU curves are complete.

### M3 — iNaturalist Birds open set

Download only after M0; hash/freeze class lists and image manifests; establish balanced multi-class baseline, controlled 10:1/50:1 long-tail and calibrated rejection.

### M4 — optional WSL2 GPU smoke test

One reproducible correctness/performance smoke test. Failure is documented and does not block release.

### M5 — CI/CD and release

Add CPU CI for distance/index correctness, split contracts, self-match exclusion, threshold logic and API health. GPU/data-scale runs remain manual or scheduled. Publish a new version only after all manifests, configs, metrics and resource logs are hashed.

## 8. Proposed first implementation order

1. Review this plan and confirm the datasets and ratios.
2. Inspect/download only the SOP mirror metadata and verify official split availability.
3. Install and smoke-test native `faiss-cpu`.
4. Implement manifest hashing and self-match assertions before embedding.
5. Run small exact/HNSW/IVF-PQ CPU tests with RSS/timing instrumentation.
6. Download/prepare SIFT1M and complete the million-vector benchmark.
7. Only then decide whether the 5.5 GB iNaturalist Birds subset is worth adding.

No GLDv2 or iNaturalist 2018 full download is part of this plan.
