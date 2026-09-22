# Seeing Similarity — STAT6207 Assignment 1

**Active workspace: Version 2.0 release.** The frozen Version 1.0 submission and its checksums remain unchanged in `releases/1.0/`. Root-level `report.pdf`, `site/` and `data/` remain the Version 1.0 assignment result; the Version 2.0 extension is documented in `docs/report/V2-RESULTS.md` with machine-readable artifacts under `results_v2/`.

Start with `report.pdf`; serve `site/` to explore the Version 1.0 result. Active documentation is indexed in `docs/README.md`. Earlier reports, scripts and submission archives remain under `backup/`.

## Main files

- `report.pdf` / `docs/report/report_text.md`: current report, organized as A1–A5 Basic Tasks, B1–B3 Advanced Tasks, and C OpenCode decisions/reproducibility.
- `site/`: portable static website and matching downloadable report.
- `data/`: locked manifests/images, embeddings, results, audit, figures and OOD supplement.
- `data/results/report_review/`: current rendered pages and layout checks; prior PDF renders live beside their reports under `backup/`.
- `docs/`: course material, protocols, plans, report notes, release notes and session records.
- `releases/1.0/Assignment1-submission.zip`: immutable Version 1.0 submission package.
- `backup/`: old versions retained for provenance, not active execution.

## Project structure

```text
src/retrieval/   Reusable encoders and distance functions
pipelines/       Data preparation, evaluation, statistics, audits and supplements
reporting/       Figures, Grad-CAM, static-site export and PDF generation
checks/          Numerical, browser, delivery and rendered-report checks
tools/           Packaging, deployment and session-export utilities
docs/            Categorized active documentation and collaboration record
data/            Locked inputs, embeddings, results and generated figures
data_v2/         Version 2.0 local datasets, immutable manifests and embeddings; raw images are not release artifacts
results_v2/      Version 2.0 machine-specific gates and benchmark measurements
configs/v2/      Preregistered Version 2.0 parameters
site/            Portable static website
releases/1.0/    Frozen Version 1.0 archive; never modified by active scripts
releases/2.0/    Version 2.0 public package, manifest, checksums and release notes
backup/          Historical development versions
```

Run Python entry points from the project root with module syntax, for example `python -m checks.verify`. This keeps imports stable after the source reorganization.

## Environment and preview

```powershell
conda env create -f environment.yml
conda activate stat6207-a1
python -m pip install -r requirements.txt
python -m http.server 8001 --directory site
```

Open http://localhost:8001. The environment pins CUDA 12.8 PyTorch wheels supporting RTX 5070 sm_120. First-time dataset/model downloads need internet. The website itself uses bundled assets and precomputed data.

## Version 2.0 execution

Version 2.0 is isolated from the frozen Version 1.0 result. It writes downloaded/local-only data to `data_v2/` and measurements to `results_v2/`; it does not modify `releases/1.0/`. The authoritative plan is `docs/plans/PLAN-2.0-ANN-OPENSET.md`, and completed gates are summarized in `docs/report/V2-EXECUTION-LOG.md`.

Core commands, in milestone order:

```powershell
python -m pipelines.v2_gpu_gate
python -m pipelines.v2_ann_smoke
python -m pipelines.v2_prepare_sop
python -m pipelines.v2_encode_sop
python -m pipelines.v2_benchmark_sop
python -m pipelines.v2_prepare_sift1m
python -m pipelines.v2_benchmark_sift1m
python -m pipelines.v2_benchmark_sift1m_scales
python -m pipelines.v2_prepare_inat
python -m pipelines.v2_freeze_long_tail
python -m pipelines.v2_acquire_inat
python -m pipelines.v2_encode_inat
python -m pipelines.v2_train_inat
python -m pipelines.v2_open_set_inat
python -m pipelines.v2_monitor_inat
python -m checks.v2_ci
python -m checks.v2_verify
python -m reporting.v2_results
python -m reporting.v2_inat_results
# after all checks pass and VERSION is 2.0:
python -m tools.package_v2
```

For the lightweight health/status endpoint, run `python -m src.service.api --port 8080` and request `/health` or `/v1/status`. It does not load embeddings or expose local dataset paths.

SOP semantic quality is evaluated on the official 60,502-image test split with exact image-ID self exclusion. SIFT1M supplies separate million-vector ANN engineering evidence. SOP and iNaturalist source images are local-only and must not be included in a public release; public artifacts are limited to code, manifests/hashes, aggregate metrics and permitted derived outputs.

## Reproduce the current experiment

The included images and embeddings reproduce the fixed results directly:

```powershell
python -m pipelines.evaluate
python -m pipelines.analyze
python -m pipelines.label_audit
python -m reporting.export_site
python -m reporting.visualize
python -m pipelines.supplement
python -m reporting.explain
python -m reporting.export_heatmaps
python -m checks.verify
python -m reporting.make_report
python -m reporting.export_site
python -m pipelines.ood
python -m checks.verify
python -m checks.browser_check
python -m checks.delivery_check
python -m checks.review_report
python -m tools.package_submission
```

Evaluation checks protocol/image hashes before using caches. For a fresh GPU encoding run, archive `data/embeddings/` and use an empty cache directory. Data preparation from scratch uses `python -m pipelines.prepare_data` in a separate project copy whose `data/` retains only `provenance/`; it refuses to overwrite `data/protocol.json`. `data/provenance/pilot_manifest.csv` supplies pilot exclusions, and 20 original Animals-10 images supply the supplementary probes, so current scripts do not depend on `backup/`.

The current immutable protocol is `data/protocol.json`, with SHA-256 in `data/protocol.sha256`. There are 1,000 reference-pool images, 500 balanced tests, six independent nonclean variants per test and 4,500 model inputs/encoder. Common preprocessing is EXIF/RGB plus direct bicubic ImageOps.fit to 224, then each official processor. Ten seeds share nested 10/25/50/100-per-class libraries. Main KNN is fixed k=5. No model champion is selected.

`reporting.visualize` regenerates current retrieval, UMAP, k and confusion figures; `reporting.explain` uses CUDA for current-data pairwise ResNet Grad-CAM. Other locked result figures are also regenerated there. `reporting.make_report` updates both root and website copies of the PDF. Refresh the session export with `python -m tools.export_session --cli PATH_TO_OPENCODE`.

`reporting.export_heatmaps` exports all ResNet-18 query/condition/distance nearest-five and farthest-five pairwise Grad-CAM maps to `site/heatmaps/`. The layer4 maps exploit the exact global-average-pooling gradient identity and are checked against autograd for both sides of 21 condition/metric pairs. Compact 7×7 maps are interpolated over the actual model crop in the browser. Targets are cosine similarity or negative L1/L2 distance, never the KNN vote. DINOv2/CLIP do not display ResNet maps as their own attribution.

The main explorer shows an explicit updated-configuration line, all k voting neighbours, pair-selection buttons and optional Grad-CAM. The unseen/error section independently supports all ten preselected cases and four displayed errors. UMAP includes source-class/split legends, a selected-query ring and observations about the actual projection. Changing k need not change the top-five rankings or predicted class; this is made explicit in the interface.

Distance labels distinguish **L1/L2 (raw features)** from **cosine (unit directions)**. `raw` means no extra embedding normalization before the distance function; cosine still normalizes internally. `unit` explicitly normalizes both query and reference embeddings. Only unit-L2 and cosine are guaranteed to have the same ordering, since squared unit-L2 equals twice cosine distance. Pixel standardization and model LayerNorm are different operations. `checks.verify` independently checks this identity and full rankings using float64/SciPy for all 21 model/condition combinations at seed 1001, and writes `data/results/normalization_check.json` plus its website copy. This diagnostic feeds the website explanation and report B2; run it before generating the report.

## Results and verification

`pipelines.supplement` adds a post-hoc descriptive raw RGB pixel KNN baseline on the locked clean test and all ten shared reference draws (224×224, Euclidean, k=5). It runs float64 distances on CUDA, verifies sampled distances by direct subtraction, and saves predictions and `data/results/supplement.json`, including class-specific ResNet failure counts. Run before `reporting.make_report`. The extra dog retrieval uses the first dog in the existing preselected display list.

The prespecified cosine six-perturbation endpoint is 94.62% for ResNet-18, 98.25% for DINOv2 and 98.2733% for CLIP. The latter two each exceed ResNet (Holm p≈0.000300); DINOv2 versus CLIP is not clearly different (p≈0.9566). This is not equivalence. Within CLIP, L1 exceeds L2/cosine by about 0.42/0.39 percentage points. No cost ranking or CPU/GPU speed ratio is claimed.

All labels are source-provided and unverified. Independent EfficientNet flags 223 agreements, 276 uncertain images and one high-confidence disagreement; none changes the lock. Query bootstrap is paired and source-class stratified, conditional on ten realized reference draws. Labels, public pretraining overlap and synthetic perturbations limit the claims.

Checks are recorded in `data/results/`: 111 metric records, 63 independently checked distance configurations, paired tests, exact regeneration of 3,000 variants, actual Edge interactions/mobile layout and PDF rendering. See `docs/report/REPORT-REVISION.md` for the visual review and restored coverage.

## Publish

```powershell
hf auth login
python -m tools.deploy --repo YOUR_USERNAME/stat6207-seeing-similarity
```

Deploys only `site/` to a Hugging Face Static Space. An authenticated user account is required; no public deployment is claimed.

## Sources

- PetImages v1: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
- Animals-10 v2: https://www.kaggle.com/datasets/alessiocorrado99/animals10
- DINOv2: https://huggingface.co/facebook/dinov2-small
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32
- torchvision ResNet18 IMAGENET1K_V1 / EfficientNet_B0 IMAGENET1K_V1. ResNet-50 was not downloaded.
