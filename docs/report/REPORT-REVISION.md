# Report comparison and revision record

## Comparison baseline

Both preceding PDFs were extracted and rendered page by page:

- `backup/pilot/report.pdf`: 7 pages, 17,158 extracted characters; richer qualitative figures but superseded experiment and invalid winner/cost conclusions.
- `backup/v3-before-reorganization/report.pdf`: 6 pages, 11,124 extracted characters; valid locked experiment, but omitted important evidence and had a mostly empty references page.

Page count or text length alone does not establish completeness. Comparing figures and sections confirmed excessive removal in the first v3 report: the PDF lacked a dedicated top-five/bottom-five retrieval illustration, UMAP, numerical normalization comparison, pairwise attribution and detailed current-case explanations. k sensitivity and confusion matrices were calculated but not adequately presented. References and course connections were overly compressed. These omissions matter to both basic rubric coverage and interpretability.

## Restored using current experiment evidence

- A2: nearest/farthest montage for the first preselected display query, with distances and IDs.
- A3: clean metrics and mean confusion matrices, with explicit vote/sampling rules.
- A4: ten unseen images, systematic failure gallery and individual observations; no claim of causal or human label verification.
- A5: website controls, three UMAP panels and descriptive OOD scope.
- B1: encoder rationale, clean uncertainty, sample efficiency, robustness, paired model tests and inference assumptions.
- B2: explicit distance formulas, raw/unit accuracy table, P@5, stress endpoints and all nine within-encoder comparisons.
- B3: k sensitivity values/figure, strengths/weaknesses, STAT6207 geometry/local decision regions, current-data GPU pairwise Grad-CAM and limitations.
- C: collaboration decisions, hashes, reproduction/checks and expanded citations.

The current report uses A1–A5 for Basic Tasks 1–5 (50%), B1–B3 for Advanced Tasks 1–3 (50%), with repeated labels marked by descriptive continuation titles. PDF bookmarks follow these sections. Figures and captions stay together; tables repeat headers. Explicit page breaks separate reviewable task units.

Historical validation-selected champions, old accuracy results, cost tables and CPU/GPU ratios remain only in backup. They are not restored as current evidence. The locked test bytes, source labels, protocol and primary statistics are unchanged by file reorganization or report additions.

## Visual review evidence

### Retrieval and pixel-baseline supplement

Current report: 11 pages, 23,931 extracted characters. A2 adds dog_0525, the first dog from the existing fixed display list, as Figure 1b; A3 moves to a separate page for legibility. A4 states seed-1001 class error counts (cats 7/250, dogs 2/250) and ten-seed mean class rates. A5 explicitly directs readers to site/ood.html. B3 reports the post-hoc clean RGB-pixel Euclidean k=5 baseline (52.12% mean accuracy), compared descriptively with the same raw-L2 rule for frozen features. CUDA float64 distances were spot-checked by direct subtraction; independent SciPy voting was checked for 50 query/seed combinations. Enlarged pages 2, 4, 5 and 9 were visually inspected; text-bound, rubric and bookmark checks passed.

### Normalization clarification

The latest B2 revision defines raw/unit consistently with the code: L1/L2 use raw embeddings in the main experiment, while cosine normalizes internally. It explains the unit-L2/cosine identity, distinguishes pixel preprocessing and LayerNorm from embedding normalization, and reports independent float64 checks on all 21 encoder/condition combinations. Full rankings agree after normalization; raw ordered top-five lists can differ. Formal results and the protocol remain locked. The updated PDF contains 10 pages and 22,664 extracted characters. All pages were checked via contact sheets and B2 at enlarged scale; tables and explanatory text fit without clipping. Desktop and 390-pixel website screenshots confirm the expanded normalization guide is readable.

`data/results/report_review/` holds current 150%-scale page renders, four-page contact sheets, extracted text and `layout_check.json` with all three versions' counts. Earlier PDF renders are kept under each version's `backup/.../report_review/` directory. Automated checks include page text bounds and expected A1–A5/B1–B3 sections; rendered pages are also inspected for clipping, unreadable graphics, isolated captions and unexpected empty pages. The current report is 10 pages, with task-sized page groups rather than a references-only final page. Its 21,757 extracted characters reflect restored evidence and explanation, not recycled pilot results.
