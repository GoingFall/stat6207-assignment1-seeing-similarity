# Seeing Similarity

STAT6207 Applied Deep Learning | Assignment 1 | September 2026

Frozen image representations for retrieval and KNN. This report follows the assignment numbering: A1-A5 are the five Basic Tasks (50%); B1-B3 are the three Advanced Tasks (50%). C records OpenCode decisions and reproducibility. Experimental protocol v3 is unchanged by this report reorganization.

Requirement | Evidence in this report

A1 Download images and encode with ResNet | Dataset, automatic audit and ResNet features

A2 Five most similar and dissimilar images | Fixed-query nearest/farthest retrieval figure

A3 KNN classifier | Voting rule, performance and confusion matrices

A4 Ten unseen images and failed cases | Preselected display and systematic error inspection

A5 Website | Static explorer, UMAP and OOD supplement

B1 Three encoding models | Matched clean and perturbation comparisons

B2 Three Lecture 1 distances | L1/L2/cosine, normalization and paired tests

B3 Strengths/weaknesses and STAT6207 | k sensitivity, feature geometry and attribution


## A1. Download images and encode with ResNet

Kaggle PetImages v1 (bhavikjikadara/dog-and-cat-classification-dataset) supplies a 1,000-image reference pool (500 cats, 500 dogs) and 500 locked tests (250/class). Each largest evaluated library uses 100 cats and 100 dogs, satisfying the minimum requirement. Twenty non-cat/dog Animals-10 v2 probes remain separate. There is no validation set or performance-selected default.

Before inference, 3,817 decodable candidates (fixed 1,800/class draws plus pilot images) were audited: 8 pHash candidate pairs and 0 decode failures. File/pixel hashes and pHash Hamming <=10 form conservative groups, supplemented by ORB matching; one representative/group is retained and pilot-connected groups are excluded. This audits the candidate pool, not the full source, and can miss transformed duplicates or merge unrelated images.

All inputs undergo EXIF correction, RGB conversion and direct bicubic center-fit to 224 x 224, then the official encoder processor. This executed canvas differs from the historical short-edge-256 proposal and was recorded in the pre-inference lock. ResNet-18 uses ImageNet weights, replaces its classifier with Identity, and returns a frozen 512-D pooled vector. KNN uses these vectors without training a neural classifier.

Labels remain source_unverified. Independent EfficientNet-B0 flags 223 agreements, 276 uncertain images and one high-confidence disagreement; no image or label changes. Public-source/pretraining overlap cannot be excluded. “Unseen” means excluded from the experimental reference library and pilot, not proven absent from pretraining. Detailed audit rules and hashes are retained with the data.


## A2. Five most similar and dissimilar images

The query is cat_0525, the first ID in the ten-image display list fixed before inference (seed 42003). It is compared with 200 clean references from seed 1001 under ResNet-18 cosine distance. Lower distance means more similar. The illustration is fixed by the display protocol, not chosen for the most persuasive retrieval result.

Figure 1a. Cat query, five smallest distances (middle) and five largest distances (bottom). Distances are computed in 512-D space; source labels do not enter ranking.

Figure 1b. Additional query dog_0525: the first dog in the same preselected display list. Query is repeated at left; nearest five above, farthest five below. Added for class coverage after report review, without choosing by retrieval quality; same encoder, metric and library as Figure 1a.


## A3. Build a KNN classifier

For each query, sort distances to labelled references, take the first k=5, and predict the majority class with uniform votes. Stable image-ID order resolves exact distance ties. Odd k avoids a binary vote tie. Test queries never vote for each other. Reference sizes are 10/25/50/100 per class, nested within each seed; ten independent shuffled draws (1001-1010) may overlap between seeds. All encoders share these indices.

ResNet-18/cosine at 100/class gives mean clean accuracy 97.64% (95% query-bootstrap interval [96.62, 98.54]%), macro-F1 0.9764, P@5 0.9553, and reference-seed SD 0.57 percentage points. Accuracy equals balanced accuracy because tests are class-balanced. P@5 is the fraction of five neighbours matching the source label; it is not five independent classification outcomes.

Figure 2. Clean cosine k=5 confusion matrices, averaged over ten reference draws. Fractional entries are mean counts, not additional independent test observations.


## A4. Ten unseen images and failed cases

Five source-labelled cats and five dogs were sampled for display before inference. Predictions below use ResNet-18/cosine/k=5, 100/class, seed 1001; all aggregate conclusions use the complete 500-image test. This teaching setting is not a selected champion.

Figure 3. All ten preselected display images, with source label and KNN prediction.

Figure 4. First four clean source-label disagreements by image ID for the same setting. Each row contains the query and its five supporting neighbours.

Full-test class failure rates for Figure 4 (seed 1001): cat 7/250 (2.80%); dog 2/250 (0.80%). Across ten draws, mean class failure rates are cat 3.40%; dog 1.32%. The first-four display contains only cats because IDs are class-sorted; it is not the class distribution of all failures. Rates measure source-label disagreements.

cat_0566 is a stretched orange cat on pale upholstery; four neighbours are dogs, several sharing warm colours or soft furnishing. cat_0649 is a small dark cat in a cluttered scene; four dark-dog neighbours outvote the first cat neighbour. cat_0652 has an oblique, low-detail face; three dogs win a 3:2 vote. cat_0673 is viewed close-up through bars; four dark-dog neighbours win 4:1. Appearance/background similarities are plausible nuisance cues, not proof of the model's causal mechanism. These are observations on displayed images, not human label verification.


## A5. Build a website to show the results

The deliverable is site/index.html with local HTML/CSS/JavaScript, bundled Plotly, images and precomputed evidence. Serve site/ over HTTP; no notebook or model server is needed for browsing. report.pdf is available both at the project root and from the website. The initial configuration is ResNet-18/cosine/k=5, seed 1001, 100 references/class.

Control / panel | What reviewers can inspect

Query, encoder and distance | 500 tests; ResNet-18, DINOv2, CLIP; L1/L2/cosine

Condition and exploratory k | Clean plus six independent variants; k=1/3/5/7/9

Nearest/farthest and vote | Five neighbours at each extreme; actual distances and cat/dog counts

Results and maps | 111 prescribed metric records, 12 paired comparisons, query-click UMAP

Separate OOD page | 20 Animals-10 queries; forced binary predictions and nearest neighbours

Figure 5. Clean cosine-UMAP panels. Fit on 200 seed-1001 references, transform 500 tests; n_neighbors=15, min_dist=0.1, random_state=42. Circles: references; stars: tests. Colours: source classes.

The UMAP axes are unitless and independently fitted per encoder; panel positions cannot be aligned or used to rank model quality. Projection can distort both local and global geometry. KNN and retrieval always use the original embeddings. The website maps remain clean even when a perturbed query is selected, and exploratory k does not alter the official fixed-k results.

OOD queries are non-cat/dog Animals-10 examples (horse, sheep, cow and elephant). Binary KNN must still output cat or dog, even with unanimous votes. They remain outside reference voting and all formal accuracy or inference. This supplement illustrates label-space mismatch; it is not an independently calibrated OOD detector.

Open site/ood.html, or follow the website navigation link “20 OOD probes”, to inspect all twenty probes and their nearest neighbours.

Verification uses actual Edge interactions, nearest/farthest cards, map clicks, all loaded image resources and a 390-pixel mobile viewport. Numerical checks compare exported distances with SciPy and website votes with recorded predictions. Hugging Face Static Spaces is the deployment target; public upload requires the user's authenticated account.


## B1. Implement and compare three encoders

Encoder | Feature / pretraining | Role in comparison

ResNet-18 | 512-D pooled CNN; ImageNet supervision | Convolutional baseline

DINOv2-S/14 | 384-D CLS; visual self-supervision | Transferable visual representation

CLIP-B/32 | 512-D projected image vector; image-text contrastive | Language-aligned visual representation

All weights are frozen. DINOv2 uses facebook/dinov2-small; CLIP uses openai/clip-vit-base-patch32 with vision projection, without text prompts. Architecture, pretraining objective, data and processor differ simultaneously, so this compares complete pretrained systems rather than isolating a causal architecture effect. ResNet-50 remains future work and was not downloaded.

Clean cosine, k=5 | Accuracy % [95% CI] | Macro-F1 | P@5 | Seed SD pp

resnet18 | 97.64 [96.62, 98.54] | 0.9764 | 0.9553 | 0.57

dinov2 | 98.68 [97.86, 99.38] | 0.9868 | 0.9770 | 0.47

clip | 99.20 [98.40, 99.80] | 0.9920 | 0.9875 | 0.21

Figure 6. Clean sample efficiency, cosine and k=5. Means span ten reference draws; bars show seed SD, not query confidence intervals.

resnet18: increasing the reference library from 10 to 100/class changes clean accuracy from 95.40% to 97.64%, and P@5 from 0.8998 to 0.9553.

dinov2: increasing the reference library from 10 to 100/class changes clean accuracy from 94.56% to 98.68%, and P@5 from 0.8752 to 0.9770.

clip: increasing the reference library from 10 to 100/class changes clean accuracy from 98.70% to 99.20%, and P@5 from 0.9290 to 0.9875.

These clean/sample-size results are descriptive; no default is chosen from their maxima. P@5 spans 25% of a 20-image library but only 2.5% of a 200-image library, so within-size retrieval comparisons have clearer meaning than interpreting its cross-size trend as representation quality alone. Neighbours and repeated reference seeds are not independent test samples.


## B1. Encoder comparison under fixed perturbations

Three independent tracks each share the same clean baseline: Gaussian blur radius 1/2; JPEG quality 50/20 with 4:2:0 subsampling; grey square occlusion of side 71/112 pixels on the 224-pixel canvas. Occlusion positions are deterministic from seed 42004 and image ID. Variants are generated from clean images, never stacked; references remain clean. Severity was not adjusted after inspecting results. No human recognizability claim is made.

Figure 7. Cosine, k=5 and 100/class. Shading is the 95% paired-query interval; all models share the same original queries, variants and ten reference draws.

Cosine encoder | Six-condition endpoint accuracy %

resnet18 | 94.62

dinov2 | 98.25

clip | 98.27

Paired comparison | Difference pp | 95% CI pp | Holm p

resnet18|cosine minus dinov2|cosine | -3.630 | [-4.79, -2.53] | 0.0003

resnet18|cosine minus clip|cosine | -3.653 | [-4.83, -2.62] | 0.0003

dinov2|cosine minus clip|cosine | -0.023 | [-0.69, 0.61] | 0.9566

DINOv2 and CLIP each exceed ResNet-18 by about 3.6 percentage points on the prespecified six-perturbation endpoint. DINOv2 minus CLIP is -0.023 percentage points, with interval [-0.693, 0.610] and Holm p=0.9566. This is insufficient evidence of a difference, not proof of equivalence. These conclusions concern source labels and this fixed synthetic mixture, not a universal ranking.

Paired inference and its scope

For each original query, correctness is averaged over six nonclean conditions and ten fixed reference draws. A source-class-stratified paired query bootstrap uses 5,000 draws (seed 42005), keeping all variants/models together. Macro-F1 is recomputed per draw. Intervals condition on the realized reference pool/draws; seed SD is reported separately. Degenerate intervals at a ceiling do not establish perfect population performance.

Two-sided paired sign permutations use 10,000 draws (seed 42006), a +1 Monte Carlo correction, and an exchangeability assumption. Holm correction is separate for three encoder pairs at cosine and nine within-encoder distance pairs. Individual tracks, clean scores, P@5, reference-size and k curves are descriptive. Intervals shown here are marginal 95% intervals, not simultaneous Holm-adjusted intervals.


## B2. Compare three Lecture 1 distances

The user confirmed that “in L1” means Lecture 1. Implemented distances for vectors x and y are: Manhattan L1 = sum_i |x_i-y_i|; Euclidean L2 = sqrt(sum_i (x_i-y_i)^2); cosine = 1 - (x dot y)/(||x|| ||y||). Lower means more similar; zero-norm protection is included. Distance magnitudes are not directly comparable across encoders or metrics.

The website and primary comparisons use raw embeddings for L1/L2; cosine normalizes internally. Raw means no extra normalization before the distance function, not that cosine retains magnitude. Unit means both query and reference embeddings are explicitly divided by their L2 norms. Pixel standardization and model LayerNorm do not ensure unit-length output; the extracted CLIP vectors are also non-unit.

Raw L2 obeys ||x-y||^2 = ||x||^2 + ||y||^2 - 2||x||||y||cos(theta), so varying reference lengths can change its ranking relative to cosine. For nonzero unit vectors u=x/||x|| and v=y/||y||, ||u-v||^2 = 2(1-u dot v). Thus unit L2 and cosine have identical ordering, apart from numerical ties; their agreement is a geometric identity, not independent performance evidence.

Encoder / metric | Clean raw % | Clean unit % | Raw P@5 | Stress raw %

resnet18/l1 | 97.40 | 97.76 | 0.9569 | 94.36

resnet18/l2 | 97.06 | 97.64 | 0.9522 | 94.31

resnet18/cosine | 97.64 | 97.64 | 0.9553 | 94.62

dinov2/l1 | 98.64 | 98.52 | 0.9762 | 98.11

dinov2/l2 | 98.66 | 98.68 | 0.9765 | 98.23

dinov2/cosine | 98.68 | 98.68 | 0.9770 | 98.25

clip/l1 | 99.32 | 99.30 | 0.9894 | 98.67

clip/l2 | 99.22 | 99.20 | 0.9875 | 98.24

clip/cosine | 99.20 | 99.20 | 0.9875 | 98.27

Normalization ablation is limited to clean, 100/class, k=5. Stress values average six nonclean conditions and ten seeds. All nine configurations are shown; unit comparisons remain descriptive. The maximum unit-L2-squared versus twice-cosine numerical discrepancy is 2.1e-06.

Independent float64 verification on seed 1001: all 21 encoder/condition configurations (500 queries x 200 references each) have zero full-ranking mismatches after normalizing both sides; maximum identity error 5.3e-15. For raw L2 versus cosine, ordered top-five lists differ on resnet18 485/500, dinov2 306/500, clip 419/500 clean queries. This need not change the neighbour set or KNN prediction. It diagnoses geometry; formal float32 results remain as locked. See normalization_check.json.

Within-encoder comparison | Difference pp | 95% CI pp | Holm p

resnet18|l1 minus resnet18|l2 | +0.053 | [-0.23, 0.35] | 1.0000

resnet18|l1 minus resnet18|cosine | -0.257 | [-0.67, 0.13] | 0.8631

resnet18|l2 minus resnet18|cosine | -0.310 | [-0.66, 0.03] | 0.5141

dinov2|l1 minus dinov2|l2 | -0.120 | [-0.24, -0.02] | 0.2863

dinov2|l1 minus dinov2|cosine | -0.137 | [-0.30, 0.01] | 0.5141

dinov2|l2 minus dinov2|cosine | -0.017 | [-0.13, 0.10] | 1.0000

clip|l1 minus clip|l2 | +0.423 | [0.25, 0.61] | 0.0009

clip|l1 minus clip|cosine | +0.393 | [0.23, 0.56] | 0.0009

clip|l2 minus clip|cosine | -0.030 | [-0.13, 0.07] | 1.0000

Within CLIP, raw L1 improves the endpoint over L2 by 0.423 percentage points and over cosine by 0.393 points (Holm p=0.0009 each). The differences are statistically detectable but small. No other distance pair passes its nine-test Holm family. An unadjusted interval excluding zero can coexist with a nonsignificant corrected test; this is not a contradiction.


## B3. Strengths, weaknesses and STAT6207 concepts

Representation-based transfer learning

Frozen pretrained features transfer a source-domain representation into a small labelled target memory. A post-hoc descriptive baseline now compares flattened 224 x 224 RGB pixels in [0,1] (150,528 coordinates), raw Euclidean distance, uniform k=5, the same 500 clean queries and ten 100/class reference draws. It uses the locked canvas without a model processor, float64 GPU distances and stable image-ID ties; no settings were tuned.

Raw-pixel KNN: accuracy 52.12%, seed SD 2.86 pp, macro-F1 0.4974, P@5 0.5152. For the same raw-L2 rule, ResNet-18/DINOv2/CLIP accuracies are 97.06/98.66/99.22%. This provides empirical support for pretrained representations over this pixel baseline on this dataset. It is not a prespecified significance test or a controlled isolation of representation learning from pretraining and processing; it does not cover perturbations.

System | Observed strength | Observed limit / interpretation

ResNet-18 | 97.64% clean cosine accuracy with a simple frozen baseline | 94.62% stress endpoint, below the other two; displayed errors are consistent with nuisance appearance similarity

DINOv2 | 98.25% cosine stress endpoint; stable semantic neighbourhoods in this test | No clear corrected difference from CLIP; this does not isolate the effect of self-supervision

CLIP | 99.20% clean cosine accuracy; L1 adds a small stress benefit | Cosine endpoint 98.27%, not demonstrably above DINOv2; no text prompts or open-set recognition evaluated

Local decision regions and the role of k

KNN creates local, generally nonlinear decision regions in feature space. Small k follows individual neighbours and may be sensitive to noise; larger k averages votes but can cross a class boundary. Increasing k is not guaranteed to improve accuracy. The locked main rule remains k=5; the clean/cosine/100-class sensitivity analysis does not select a new setting.

Figure 8. Descriptive k sensitivity averaged over ten seeds. Only this clean/cosine configuration is prescribed for the k analysis.

Encoder | k=1 % | k=3 % | k=5 % | k=7 % | k=9 %

resnet18 | 96.10 | 97.12 | 97.64 | 97.76 | 97.56

dinov2 | 98.50 | 98.72 | 98.68 | 98.50 | 98.48

clip | 99.04 | 99.16 | 99.20 | 99.24 | 99.28

Cosine corresponds to normalized inner-product geometry; removing magnitude helps only if magnitude is nuisance rather than useful signal. L1 remains dependent on the coordinate basis, whereas orthogonal rotations preserve L2 and cosine. None of these metrics automatically removes semantic confounding from learned features.


## B3. Inspect similarity evidence and limitations

Figure 9. Current-data ResNet-18 pairwise cosine Grad-CAM for the first preselected display query. Nearest pair on top, farthest pair below; query/candidate are differentiated separately with the other embedding held fixed.

The target is pairwise cosine similarity, not a class logit. Gradients at ResNet layer4 are spatially averaged to weight feature maps, followed by ReLU and interpolation. The overlay highlights positive contributions to similarity, not all evidence for dissimilarity. Per-map normalization means colour intensity cannot be used to compare scores across pairs. KNN sorting/voting is not differentiated. These maps are neither Transformer attention nor complete causal explanations; attribution is implemented only for ResNet and recomputed on GPU for the current data.

Residual uncertainty beyond statistical intervals

Automatic label flags do not establish verified truth. The audit sums ImageNet cat indices 281-285 and domestic dog indices 151-268; opposite probability mass >=0.8 with source mass <=0.1 flags disagreement. Low mass is uncertain, not necessarily incorrect. EfficientNet may share biases or pretraining images with the compared models. All 500 labels and images remain fixed, so filtering cannot inflate the headline scores.

Occlusion can remove decisive object evidence, and the centre-fit canvas can crop it before perturbation. Preset mild/moderate severity indicates only relative synthetic strength. A high score under this design does not establish natural-corruption robustness, human recognizability, pretraining independence, or calibrated OOD detection. Query-bootstrap intervals and Holm correction cannot repair these systematic limitations.

Failure montages and maps support inspection but are not causal tests. The first-four-error display overrepresents cats because IDs are sorted by class; all-source-class errors and every seed remain in predictions.npz. The ten demonstration images illustrate the assignment requirement, while the full 500 tests support quantitative conclusions. Future work could use independently verified external data, calibrated unknown-class rejection and a controlled architecture/pretraining comparison.


## C. OpenCode decisions and reproducibility

Decision stage | Action and rationale

Pilot and user review | The 270-image pilot reached a fixed-k classification ceiling. User feedback rejected arbitrary tie-break champions and confounded cost claims.

Prospective revision | Lock 1,000 references / 500 tests, ten draws and independent perturbations after seeing the pilot, before new inference. This is not pre-pilot preregistration.

No human label review | Replace human review with reproducible decoding/hash/group audits plus independent label flags; evaluate against unchanged source labels.

Environment and scope | Dedicated stat6207-a1 environment; CUDA on RTX 5070; frozen ResNet-18/DINOv2/CLIP; no ResNet-50 download.

Delivery revision | Current report/code/site at root and standard paths; historical versions in backup/. Restore retrieval, maps, normalization, attribution and case discussion using current data.

Follow README.md for reproduction order, including `python -m reporting.export_site`, `python -m reporting.export_heatmaps` and `python -m checks.verify` before `python -m reporting.make_report`. Data preparation is only for a fresh data directory retaining data/provenance/ and refuses to overwrite an existing lock. Provided locked images and embeddings allow result regeneration without new encoding. Image hashes are checked before evaluation; moving the source code does not change protocol or test-manifest bytes.

Verification: verify.py independently checks 111 metric records, 63 exported distance configurations against SciPy, votes, balanced nested draws, group separation and Holm families. delivery_check.py regenerates all 3,000 perturbations exactly. browser_check.py exercises the real site. review_report.py renders every PDF page, checks text bounds and retains comparison sheets for both previous reports.

Test manifest SHA-256: b825458b71b55bbbb6ec3aeedfe12bd39e5ad2484ba21e0b7523d6f5aaffa9bb

Protocol SHA-256: fd36526774dadb9d2ba0778193de0213901e4ba32683f5b2e72b1cb6e987a175

The submission contains report.pdf, site/, source scripts, environment pins, data/results and the exported OpenCode session. Launch: python -m http.server 8001 --directory site. Public Hugging Face deployment remains account-dependent. Historical CPU/GPU ratios and encoder cost tables are excluded from current conclusions.

References

[1] docs/course/Assignment 1 Instructions.txt and docs/course/Lecture 2.txt. Basic A1-A5 / Advanced B1-B3 mapping follows the assignment verbatim.

[2] He et al. (2016). Deep Residual Learning for Image Recognition. CVPR. torchvision ResNet18 IMAGENET1K_V1.

[3] Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193; facebook/dinov2-small.

[4] Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020; openai/clip-vit-base-patch32.

[5] McInnes et al. (2018). UMAP. arXiv:1802.03426. Selvaraju et al. (2017). Grad-CAM. ICCV. Tan and Le (2019). EfficientNet. ICML.

[6] Kaggle: bhavikjikadara/dog-and-cat-classification-dataset v1; alessiocorrado99/animals10 v2. Full source URLs and reproduction commands are in README.md.
