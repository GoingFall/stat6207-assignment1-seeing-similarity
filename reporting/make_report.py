"""Rubric-aligned report generated from the immutable experiment evidence."""
import json,shutil
import numpy as np
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from pipelines.prepare_data import ROOT,OUT,R1

def main():
    data=json.loads((R1/'evaluation.json').read_text());p=json.loads((OUT/'protocol.json').read_text());audit=json.loads((R1/'label_audit.json').read_text());candidates=json.loads((OUT/'audit.json').read_text())
    site=json.loads((ROOT/'site/v1.0/assets/data.json').read_text());models=p['models'];figures=R1/'figures'
    geometry=json.loads((R1/'normalization_check.json').read_text());assert geometry['status']=='passed' and geometry['protocol_hash']==data['protocol_hash']
    supplement=json.loads((R1/'supplement.json').read_text());assert supplement['status']=='passed' and supplement['protocol_hash']==data['protocol_hash']
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=13;styles['BodyText'].spaceAfter=7
    styles['Heading1'].fontSize=16;styles['Heading1'].leading=20;styles['Heading1'].textColor=colors.HexColor('#176b62')
    styles['Heading2'].fontSize=11;styles['Heading2'].leading=14
    styles.add(ParagraphStyle(name='CaptionText',fontSize=8,leading=10.5,textColor=colors.HexColor('#526762'),spaceAfter=8))
    styles.add(ParagraphStyle(name='TableText',fontSize=8,leading=10))
    styles.add(ParagraphStyle(name='SmallText',fontSize=8.5,leading=11,spaceAfter=6))
    story=[];text=[];outline=[]
    def para(s,style='BodyText'):
        story.append(Paragraph(escape(s),styles[style]));text.append(s)
    def heading(s):
        obj=Paragraph(escape(s),styles['Heading1']);obj.bookmark='section-'+str(len(outline));outline.append(s);story.append(obj);text.append('\n## '+s)
    def sub(s):para(s,'Heading2')
    def page():story.append(PageBreak())
    def table(rows,widths):
        cells=[[Paragraph(escape(str(c)),styles['TableText']) for c in row] for row in rows]
        obj=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8efea')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cbd8ce')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(obj);story.append(Spacer(1,8));text.extend(' | '.join(map(str,r)) for r in rows)
    def figure(name,caption,width=495,maxheight=400):
        w,h=ImageReader(str(figures/name)).getSize();scale=min(width/w,maxheight/h)
        story.append(KeepTogether([Image(str(figures/name),width=w*scale,height=h*scale),Spacer(1,4),Paragraph(escape(caption),styles['CaptionText'])]));text.append(caption)
    def metric(model,dist='cosine',mode='raw',k=5,condition='clean',size=100):return next(r for r in data['metrics'] if r['key']==f'{model}|{dist}|{mode}|{condition}|{size}|{k}')
    def pct(v):return f'{v*100:.2f}'
    def ci(v):return f'[{v[0]*100:.2f}, {v[1]*100:.2f}]'
    def comparison_rows(group):return [[r['a']+' minus '+r['b'],f"{100*r['difference']:+.3f}",ci(r['ci']),f"{r['holm_p']:.4f}"] for r in data['comparisons'] if r['family']==group]

    story.append(Paragraph('Seeing Similarity',styles['Title']))
    para('STAT6207 Applied Deep Learning | Assignment 1 | September 2026','CaptionText')
    para('Frozen image representations for retrieval and KNN. This report follows the assignment numbering: A1-A5 are the five Basic Tasks (50%); B1-B3 are the three Advanced Tasks (50%). C records OpenCode decisions and reproducibility. Experimental protocol v3 is unchanged by this report reorganization.')
    table([['Requirement','Evidence in this report'],['A1 Download images and encode with ResNet','Dataset, automatic audit and ResNet features'],['A2 Five most similar and dissimilar images','Fixed-query nearest/farthest retrieval figure'],['A3 KNN classifier','Voting rule, performance and confusion matrices'],['A4 Ten unseen images and failed cases','Preselected display and systematic error inspection'],['A5 Website','Static explorer, UMAP and OOD supplement'],['B1 Three encoding models','Matched clean and perturbation comparisons'],['B2 Three Lecture 1 distances','L1/L2/cosine, normalization and paired tests'],['B3 Strengths/weaknesses and STAT6207','k sensitivity, feature geometry and attribution']],[175,320])
    heading('A1. Download images and encode with ResNet')
    para('Kaggle PetImages v1 (bhavikjikadara/dog-and-cat-classification-dataset) supplies a 1,000-image reference pool (500 cats, 500 dogs) and 500 locked tests (250/class). Each largest evaluated library uses 100 cats and 100 dogs, satisfying the minimum requirement. Twenty non-cat/dog Animals-10 v2 probes remain separate. There is no validation set or performance-selected default.')
    para(f"Before inference, {len(candidates['records']):,} decodable candidates (fixed 1,800/class draws plus pilot images) were audited: {len(candidates['pairs'])} pHash candidate pairs and {len(candidates['rejected'])} decode failures. File/pixel hashes and pHash Hamming <=10 form conservative groups, supplemented by ORB matching; one representative/group is retained and pilot-connected groups are excluded. This audits the candidate pool, not the full source, and can miss transformed duplicates or merge unrelated images.")
    para('All inputs undergo EXIF correction, RGB conversion and direct bicubic center-fit to 224 x 224, then the official encoder processor. This executed canvas differs from the historical short-edge-256 proposal and was recorded in the pre-inference lock. ResNet-18 uses ImageNet weights, replaces its classifier with Identity, and returns a frozen 512-D pooled vector. KNN uses these vectors without training a neural classifier.')
    para('Labels remain source_unverified. Independent EfficientNet-B0 flags 223 agreements, 276 uncertain images and one high-confidence disagreement; no image or label changes. Public-source/pretraining overlap cannot be excluded. “Unseen” means excluded from the experimental reference library and pilot, not proven absent from pretraining. Detailed audit rules and hashes are retained with the data.','SmallText')

    page();heading('A2. Five most similar and dissimilar images')
    qi=next(i for i,r in enumerate(site['queries']) if r['id']==p['display_ids'][0]);query=site['queries'][qi]
    para(f"The query is {query['id']}, the first ID in the ten-image display list fixed before inference (seed 42003). It is compared with 200 clean references from seed 1001 under ResNet-18 cosine distance. Lower distance means more similar. The illustration is fixed by the display protocol, not chosen for the most persuasive retrieval result.")
    figure('retrieval.png','Figure 1a. Cat query, five smallest distances (middle) and five largest distances (bottom). Distances are computed in 512-D space; source labels do not enter ranking.',maxheight=250)
    dog_id=next(ident for ident in p['display_ids'] if next(r for r in site['queries'] if r['id']==ident)['source_label']=='dog')
    figure('retrieval_dog.png',f'Figure 1b. Additional query {dog_id}: the first dog in the same preselected display list. Query is repeated at left; nearest five above, farthest five below. Added for class coverage after report review, without choosing by retrieval quality; same encoder, metric and library as Figure 1a.',maxheight=190)
    page()
    heading('A3. Build a KNN classifier')
    para('For each query, sort distances to labelled references, take the first k=5, and predict the majority class with uniform votes. Stable image-ID order resolves exact distance ties. Odd k avoids a binary vote tie. Test queries never vote for each other. Reference sizes are 10/25/50/100 per class, nested within each seed; ten independent shuffled draws (1001-1010) may overlap between seeds. All encoders share these indices.')
    r=metric('resnet18');para(f"ResNet-18/cosine at 100/class gives mean clean accuracy {pct(r['accuracy'])}% (95% query-bootstrap interval {ci(r['accuracy_ci'])}%), macro-F1 {r['macro_f1']:.4f}, P@5 {r['p5']:.4f}, and reference-seed SD {pct(r['seed_sd'])} percentage points. Accuracy equals balanced accuracy because tests are class-balanced. P@5 is the fraction of five neighbours matching the source label; it is not five independent classification outcomes.")
    figure('confusion.png','Figure 2. Clean cosine k=5 confusion matrices, averaged over ten reference draws. Fractional entries are mean counts, not additional independent test observations.',maxheight=160)

    page();heading('A4. Ten unseen images and failed cases')
    para('Five source-labelled cats and five dogs were sampled for display before inference. Predictions below use ResNet-18/cosine/k=5, 100/class, seed 1001; all aggregate conclusions use the complete 500-image test. This teaching setting is not a selected champion.')
    figure('unseen.png','Figure 3. All ten preselected display images, with source label and KNN prediction.',maxheight=185)
    figure('failures.png','Figure 4. First four clean source-label disagreements by image ID for the same setting. Each row contains the query and its five supporting neighbours.',maxheight=340)
    errors=supplement['resnet_class_errors']
    para('Full-test class failure rates for Figure 4 (seed 1001): '+ '; '.join(f"{label} {r['seed1001_errors']}/{r['total']} ({pct(r['seed1001_rate'])}%)" for label,r in errors.items())+'. Across ten draws, mean class failure rates are '+ '; '.join(f"{label} {pct(r['ten_seed_mean_rate'])}%" for label,r in errors.items())+'. The first-four display contains only cats because IDs are class-sorted; it is not the class distribution of all failures. Rates measure source-label disagreements.','SmallText')
    para('cat_0566 is a stretched orange cat on pale upholstery; four neighbours are dogs, several sharing warm colours or soft furnishing. cat_0649 is a small dark cat in a cluttered scene; four dark-dog neighbours outvote the first cat neighbour. cat_0652 has an oblique, low-detail face; three dogs win a 3:2 vote. cat_0673 is viewed close-up through bars; four dark-dog neighbours win 4:1. Appearance/background similarities are plausible nuisance cues, not proof of the model\'s causal mechanism. These are observations on displayed images, not human label verification.','SmallText')

    page();heading('A5. Build a website to show the results')
    para('The deliverable is site/index.html with local HTML/CSS/JavaScript, bundled Plotly, images and precomputed evidence. Serve site/ over HTTP; no notebook or model server is needed for browsing. report.pdf is available both at the project root and from the website. The initial configuration is ResNet-18/cosine/k=5, seed 1001, 100 references/class.')
    table([['Control / panel','What reviewers can inspect'],['Query, encoder and distance','500 tests; ResNet-18, DINOv2, CLIP; L1/L2/cosine'],['Condition and exploratory k','Clean plus six independent variants; k=1/3/5/7/9'],['Nearest/farthest and vote','Five neighbours at each extreme; actual distances and cat/dog counts'],['Results and maps','111 prescribed metric records, 12 paired comparisons, query-click UMAP'],['Separate OOD page','20 Animals-10 queries; forced binary predictions and nearest neighbours']],[150,345])
    figure('umap.png','Figure 5. Clean cosine-UMAP panels. Fit on 200 seed-1001 references, transform 500 tests; n_neighbors=15, min_dist=0.1, random_state=42. Circles: references; stars: tests. Colours: source classes.',maxheight=195)
    para('The UMAP axes are unitless and independently fitted per encoder; panel positions cannot be aligned or used to rank model quality. Projection can distort both local and global geometry. KNN and retrieval always use the original embeddings. The website maps remain clean even when a perturbed query is selected, and exploratory k does not alter the official fixed-k results.')
    para('OOD queries are non-cat/dog Animals-10 examples (horse, sheep, cow and elephant). Binary KNN must still output cat or dog, even with unanimous votes. They remain outside reference voting and all formal accuracy or inference. This supplement illustrates label-space mismatch; it is not an independently calibrated OOD detector.')
    para('Open site/ood.html, or follow the website navigation link “20 OOD probes”, to inspect all twenty probes and their nearest neighbours.','SmallText')
    para('Verification uses actual Edge interactions, nearest/farthest cards, map clicks, all loaded image resources and a 390-pixel mobile viewport. Numerical checks compare exported distances with SciPy and website votes with recorded predictions. Hugging Face Static Spaces is the deployment target; public upload requires the user\'s authenticated account.','SmallText')

    page();heading('B1. Implement and compare three encoders')
    table([['Encoder','Feature / pretraining','Role in comparison'],['ResNet-18','512-D pooled CNN; ImageNet supervision','Convolutional baseline'],['DINOv2-S/14','384-D CLS; visual self-supervision','Transferable visual representation'],['CLIP-B/32','512-D projected image vector; image-text contrastive','Language-aligned visual representation']],[100,220,175])
    para('All weights are frozen. DINOv2 uses facebook/dinov2-small; CLIP uses openai/clip-vit-base-patch32 with vision projection, without text prompts. Architecture, pretraining objective, data and processor differ simultaneously, so this compares complete pretrained systems rather than isolating a causal architecture effect. ResNet-50 remains future work and was not downloaded.')
    table([['Clean cosine, k=5','Accuracy % [95% CI]','Macro-F1','P@5','Seed SD pp']]+[[m,pct(metric(m)['accuracy'])+' '+ci(metric(m)['accuracy_ci']),f"{metric(m)['macro_f1']:.4f}",f"{metric(m)['p5']:.4f}",pct(metric(m)['seed_sd'])] for m in models],[95,190,70,65,75])
    figure('sample_efficiency.png','Figure 6. Clean sample efficiency, cosine and k=5. Means span ten reference draws; bars show seed SD, not query confidence intervals.',maxheight=220)
    for m in models:
        a,b=metric(m,size=10),metric(m,size=100);para(f"{m}: increasing the reference library from 10 to 100/class changes clean accuracy from {pct(a['accuracy'])}% to {pct(b['accuracy'])}%, and P@5 from {a['p5']:.4f} to {b['p5']:.4f}.",'SmallText')
    para('These clean/sample-size results are descriptive; no default is chosen from their maxima. P@5 spans 25% of a 20-image library but only 2.5% of a 200-image library, so within-size retrieval comparisons have clearer meaning than interpreting its cross-size trend as representation quality alone. Neighbours and repeated reference seeds are not independent test samples.')

    page();heading('B1. Encoder comparison under fixed perturbations')
    para('Three independent tracks each share the same clean baseline: Gaussian blur radius 1/2; JPEG quality 50/20 with 4:2:0 subsampling; grey square occlusion of side 71/112 pixels on the 224-pixel canvas. Occlusion positions are deterministic from seed 42004 and image ID. Variants are generated from clean images, never stacked; references remain clean. Severity was not adjusted after inspecting results. No human recognizability claim is made.')
    figure('robustness.png','Figure 7. Cosine, k=5 and 100/class. Shading is the 95% paired-query interval; all models share the same original queries, variants and ten reference draws.',maxheight=210)
    table([['Cosine encoder','Six-condition endpoint accuracy %'],*[[m,pct(data['endpoint_means'][m+'|cosine'])] for m in models]],[245,250])
    table([['Paired comparison','Difference pp','95% CI pp','Holm p']]+comparison_rows('models'),[210,85,125,75])
    para('DINOv2 and CLIP each exceed ResNet-18 by about 3.6 percentage points on the prespecified six-perturbation endpoint. DINOv2 minus CLIP is -0.023 percentage points, with interval [-0.693, 0.610] and Holm p=0.9566. This is insufficient evidence of a difference, not proof of equivalence. These conclusions concern source labels and this fixed synthetic mixture, not a universal ranking.')
    sub('Paired inference and its scope')
    para('For each original query, correctness is averaged over six nonclean conditions and ten fixed reference draws. A source-class-stratified paired query bootstrap uses 5,000 draws (seed 42005), keeping all variants/models together. Macro-F1 is recomputed per draw. Intervals condition on the realized reference pool/draws; seed SD is reported separately. Degenerate intervals at a ceiling do not establish perfect population performance.','SmallText')
    para('Two-sided paired sign permutations use 10,000 draws (seed 42006), a +1 Monte Carlo correction, and an exchangeability assumption. Holm correction is separate for three encoder pairs at cosine and nine within-encoder distance pairs. Individual tracks, clean scores, P@5, reference-size and k curves are descriptive. Intervals shown here are marginal 95% intervals, not simultaneous Holm-adjusted intervals.','SmallText')

    page();heading('B2. Compare three Lecture 1 distances')
    para('The user confirmed that “in L1” means Lecture 1. Implemented distances for vectors x and y are: Manhattan L1 = sum_i |x_i-y_i|; Euclidean L2 = sqrt(sum_i (x_i-y_i)^2); cosine = 1 - (x dot y)/(||x|| ||y||). Lower means more similar; zero-norm protection is included. Distance magnitudes are not directly comparable across encoders or metrics.')
    para('The website and primary comparisons use raw embeddings for L1/L2; cosine normalizes internally. Raw means no extra normalization before the distance function, not that cosine retains magnitude. Unit means both query and reference embeddings are explicitly divided by their L2 norms. Pixel standardization and model LayerNorm do not ensure unit-length output; the extracted CLIP vectors are also non-unit.','SmallText')
    para('Raw L2 obeys ||x-y||^2 = ||x||^2 + ||y||^2 - 2||x||||y||cos(theta), so varying reference lengths can change its ranking relative to cosine. For nonzero unit vectors u=x/||x|| and v=y/||y||, ||u-v||^2 = 2(1-u dot v). Thus unit L2 and cosine have identical ordering, apart from numerical ties; their agreement is a geometric identity, not independent performance evidence.','SmallText')
    table([['Encoder / metric','Clean raw %','Clean unit %','Raw P@5','Stress raw %']]+[[m+'/'+d,pct(metric(m,d)['accuracy']),pct(metric(m,d,'unit')['accuracy']),f"{metric(m,d)['p5']:.4f}",pct(data['endpoint_means'][m+'|'+d])] for m in models for d in p['metrics']],[145,85,85,80,100])
    para('Normalization ablation is limited to clean, 100/class, k=5. Stress values average six nonclean conditions and ten seeds. All nine configurations are shown; unit comparisons remain descriptive. The maximum unit-L2-squared versus twice-cosine numerical discrepancy is '+f"{max(r['max_identity_error'] for r in data['normalization_checks']):.2g}"+'.','SmallText')
    clean_geometry=[r for r in geometry['configurations'] if r['condition']=='clean']
    para('Independent float64 verification on seed 1001: all 21 encoder/condition configurations (500 queries x 200 references each) have zero full-ranking mismatches after normalizing both sides; maximum identity error '+f"{max(r['max_identity_error'] for r in geometry['configurations']):.2g}"+'. For raw L2 versus cosine, ordered top-five lists differ on '+', '.join(f"{r['model']} {r['raw_top5_order_mismatches']}/500" for r in clean_geometry)+' clean queries. This need not change the neighbour set or KNN prediction. It diagnoses geometry; formal float32 results remain as locked. See normalization_check.json.','SmallText')
    table([['Within-encoder comparison','Difference pp','95% CI pp','Holm p']]+comparison_rows('distances'),[210,85,125,75])
    para('Within CLIP, raw L1 improves the endpoint over L2 by 0.423 percentage points and over cosine by 0.393 points (Holm p=0.0009 each). The differences are statistically detectable but small. No other distance pair passes its nine-test Holm family. An unadjusted interval excluding zero can coexist with a nonsignificant corrected test; this is not a contradiction.','SmallText')

    page();heading('B3. Strengths, weaknesses and STAT6207 concepts')
    sub('Representation-based transfer learning')
    para('Frozen pretrained features transfer a source-domain representation into a small labelled target memory. A post-hoc descriptive baseline now compares flattened 224 x 224 RGB pixels in [0,1] (150,528 coordinates), raw Euclidean distance, uniform k=5, the same 500 clean queries and ten 100/class reference draws. It uses the locked canvas without a model processor, float64 GPU distances and stable image-ID ties; no settings were tuned.')
    para(f"Raw-pixel KNN: accuracy {pct(supplement['accuracy'])}%, seed SD {pct(supplement['seed_sd'])} pp, macro-F1 {supplement['macro_f1']:.4f}, P@5 {supplement['p5']:.4f}. For the same raw-L2 rule, ResNet-18/DINOv2/CLIP accuracies are "+'/'.join(pct(metric(m,'l2')['accuracy']) for m in models)+'%. This provides empirical support for pretrained representations over this pixel baseline on this dataset. It is not a prespecified significance test or a controlled isolation of representation learning from pretraining and processing; it does not cover perturbations.','SmallText')
    table([['System','Observed strength','Observed limit / interpretation'],['ResNet-18','97.64% clean cosine accuracy with a simple frozen baseline','94.62% stress endpoint, below the other two; displayed errors are consistent with nuisance appearance similarity'],['DINOv2','98.25% cosine stress endpoint; stable semantic neighbourhoods in this test','No clear corrected difference from CLIP; this does not isolate the effect of self-supervision'],['CLIP','99.20% clean cosine accuracy; L1 adds a small stress benefit','Cosine endpoint 98.27%, not demonstrably above DINOv2; no text prompts or open-set recognition evaluated']],[85,190,220])
    sub('Local decision regions and the role of k')
    para('KNN creates local, generally nonlinear decision regions in feature space. Small k follows individual neighbours and may be sensitive to noise; larger k averages votes but can cross a class boundary. Increasing k is not guaranteed to improve accuracy. The locked main rule remains k=5; the clean/cosine/100-class sensitivity analysis does not select a new setting.')
    figure('k_sensitivity.png','Figure 8. Descriptive k sensitivity averaged over ten seeds. Only this clean/cosine configuration is prescribed for the k analysis.',maxheight=205)
    table([['Encoder','k=1 %','k=3 %','k=5 %','k=7 %','k=9 %']]+[[m]+[pct(metric(m,k=k)['accuracy']) for k in p['k_sensitivity']] for m in models],[120,75,75,75,75,75])
    para('Cosine corresponds to normalized inner-product geometry; removing magnitude helps only if magnitude is nuisance rather than useful signal. L1 remains dependent on the coordinate basis, whereas orthogonal rotations preserve L2 and cosine. None of these metrics automatically removes semantic confounding from learned features.','SmallText')

    page();heading('B3. Inspect similarity evidence and limitations')
    figure('gradcam.png','Figure 9. Current-data ResNet-18 pairwise cosine Grad-CAM for the first preselected display query. Nearest pair on top, farthest pair below; query/candidate are differentiated separately with the other embedding held fixed.',maxheight=285)
    para('The target is pairwise cosine similarity, not a class logit. Gradients at ResNet layer4 are spatially averaged to weight feature maps, followed by ReLU and interpolation. The overlay highlights positive contributions to similarity, not all evidence for dissimilarity. Per-map normalization means colour intensity cannot be used to compare scores across pairs. KNN sorting/voting is not differentiated. These maps are neither Transformer attention nor complete causal explanations; attribution is implemented only for ResNet and recomputed on GPU for the current data.')
    sub('Residual uncertainty beyond statistical intervals')
    para('Automatic label flags do not establish verified truth. The audit sums ImageNet cat indices 281-285 and domestic dog indices 151-268; opposite probability mass >=0.8 with source mass <=0.1 flags disagreement. Low mass is uncertain, not necessarily incorrect. EfficientNet may share biases or pretraining images with the compared models. All 500 labels and images remain fixed, so filtering cannot inflate the headline scores.')
    para('Occlusion can remove decisive object evidence, and the centre-fit canvas can crop it before perturbation. Preset mild/moderate severity indicates only relative synthetic strength. A high score under this design does not establish natural-corruption robustness, human recognizability, pretraining independence, or calibrated OOD detection. Query-bootstrap intervals and Holm correction cannot repair these systematic limitations.')
    para('Failure montages and maps support inspection but are not causal tests. The first-four-error display overrepresents cats because IDs are sorted by class; all-source-class errors and every seed remain in predictions.npz. The ten demonstration images illustrate the assignment requirement, while the full 500 tests support quantitative conclusions. Future work could use independently verified external data, calibrated unknown-class rejection and a controlled architecture/pretraining comparison.')

    page();heading('C. OpenCode decisions and reproducibility')
    table([['Decision stage','Action and rationale'],['Pilot and user review','The 270-image pilot reached a fixed-k classification ceiling. User feedback rejected arbitrary tie-break champions and confounded cost claims.'],['Prospective revision','Lock 1,000 references / 500 tests, ten draws and independent perturbations after seeing the pilot, before new inference. This is not pre-pilot preregistration.'],['No human label review','Replace human review with reproducible decoding/hash/group audits plus independent label flags; evaluate against unchanged source labels.'],['Environment and scope','Dedicated stat6207-a1 environment; CUDA on RTX 5070; frozen ResNet-18/DINOv2/CLIP; no ResNet-50 download.'],['Delivery revision','Current report/code/site at root and standard paths; historical versions in archive/. Restore retrieval, maps, normalization, attribution and case discussion using current data.']],[125,370])
    para('Follow README.md for reproduction order, including python -m reporting.export_site, python -m reporting.export_heatmaps and python -m checks.verify before python -m reporting.make_report. Data preparation is only for a fresh data directory retaining data/provenance/ and refuses to overwrite an existing lock. Provided locked images and embeddings allow result regeneration without new encoding. Image hashes are checked before evaluation; moving the source code does not change protocol or test-manifest bytes.','SmallText')
    para('Verification: checks.verify independently checks 111 metric records, 63 exported distance configurations against SciPy, votes, balanced nested draws, group separation and Holm families. checks.delivery_check regenerates all 3,000 perturbations exactly. checks.browser_check exercises the real site. checks.review_report renders every PDF page, checks text bounds and retains comparison sheets for both previous reports.','SmallText')
    para('Test manifest SHA-256: '+p['test_manifest_sha256'],'SmallText');para('Protocol SHA-256: '+data['protocol_hash'],'SmallText')
    para('The submission contains report.pdf, site/, the organized source packages, environment pins, data/results and the exported OpenCode session under docs/session/. Launch: python -m http.server 8001 --directory site. Public Hugging Face deployment remains account-dependent. Historical CPU/GPU ratios and encoder cost tables are excluded from current conclusions.','SmallText')
    sub('References')
    for ref in ['[1] docs/course/Assignment 1 Instructions.txt and docs/course/Lecture 2.txt. Basic A1-A5 / Advanced B1-B3 mapping follows the assignment verbatim.',
                '[2] He et al. (2016). Deep Residual Learning for Image Recognition. CVPR. torchvision ResNet18 IMAGENET1K_V1.',
                '[3] Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193; facebook/dinov2-small.',
                '[4] Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020; openai/clip-vit-base-patch32.',
                '[5] McInnes et al. (2018). UMAP. arXiv:1802.03426. Selvaraju et al. (2017). Grad-CAM. ICCV. Tan and Le (2019). EfficientNet. ICML.',
                 '[6] Kaggle: bhavikjikadara/dog-and-cat-classification-dataset v1; alessiocorrado99/animals10 v2. Full source URLs and reproduction commands are in README.md.']:para(ref,'SmallText')
    class Report(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            if hasattr(flowable,'bookmark'):
                self.canv.bookmarkPage(flowable.bookmark);self.canv.addOutlineEntry(flowable.getPlainText(),flowable.bookmark,level=0)
    def footer(c,d):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526762'));c.drawString(44,23,'STAT6207 | Assignment 1 | Source labels unverified');c.drawRightString(551,23,str(d.page))
    Report(str(ROOT/'report/v1.0/report.pdf'),pagesize=(595,842),leftMargin=44,rightMargin=44,topMargin=32,bottomMargin=39,title='Seeing Similarity - STAT6207 Assignment 1',author='STAT6207 Assignment 1').build(story,onFirstPage=footer,onLaterPages=footer)
    folder=ROOT/'docs/report';folder.mkdir(parents=True,exist_ok=True)
    (folder/'report_text.md').write_text('# Seeing Similarity\n\n'+'\n\n'.join(text),encoding='utf-8');shutil.copy2(ROOT/'report/v1.0/report.pdf',ROOT/'site/v1.0/report.pdf')
    print('Generated root report.pdf with rubric-aligned sections and current-data evidence')

if __name__=='__main__':main()
