"""Generate revision results from authoritative verified metrics."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipelines.prepare_data import ROOT
from src.training.provenance import write_json


def main():
    def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
    verification=load('results_v21/verification.json');assert verification['status']=='passed'
    tr=load('results_v21/inat/training.json');osr=load('results_v21/inat/open_set.json');mon=load('results_v21/inat/monitoring.json');lock=load('data_v21/manifests/lock.json')
    figdir=ROOT/'results_v21/figures';figdir.mkdir(parents=True,exist_ok=True)
    methods=list(tr['long_tail_summary'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric,title in zip(axes,('accuracy','tail_macro_f1'),('Test accuracy','Corrected tail macro-F1')):
        values=[tr['long_tail_summary'][m]['metrics'][metric] for m in methods]
        ax.bar(range(3),[v['mean'] for v in values],yerr=[v['std'] for v in values],capsize=4,color='#42637d')
        ax.set_xticks(range(3),['Cross entropy','Weighted CE','Balanced sampler']);ax.set_ylabel(title+' (0–1)');ax.set_xlabel('Training method');ax.set_title(title+' — five seeds, mean ± sample SD')
    fig.savefig(figdir/'long-tail.png',dpi=160);plt.close(fig)
    summary={'version':'2.1','verification':verification,'balanced':{'knn':tr['knn'],'probe':tr['balanced_probe']['test']},'long_tail':tr['long_tail_summary'],'calibration':osr['calibration'],'open_set':{k:v['test'] for k,v in osr['methods'].items()},'monitoring':{k:v['alert_rates'] for k,v in mon['evaluations'].items()}}
    write_json(ROOT/'results_v21/summary.json',summary)
    best=max(osr['methods'],key=lambda m:osr['methods'][m]['test']['auroc_ood'])
    lines=['# Version 2.1 — content-cleaned, prediction-audited revision','',
    '## Scope and protocol','',
    'This is an audit-driven revision, not a fresh unseen benchmark. V2.0 remains historical and its release ZIP is unchanged. SOP/SIFT results are inherited historical evidence and have not been rebenchmarked in 2.1. No 2.1 ANN throughput claim is made.','',
    'Original class and split memberships are retained. Conflicting-label content is quarantined in full; same-label duplicate bytes are deduplicated using fixed train/validation/test/unknown-development/unknown-test priority and image ID. Known classes use the lowest remaining IDs per split: 20 train, 8 validation, 8 test. Selection never uses predictions. This reduced common budget avoids unequal per-class counts and preserves exact 10:1 sampling (20:2). It is not directly comparable to the original 30/10/10 experiment.','',
    f"The selected corpus has {verification['image_content_unique']:,} byte-content-unique images; {len(lock['removed'])} conflicting/duplicate records were removed before budget selection. There are 1,000 known and 243/243 unknown-development/test classes. SHA-256 checks exclude byte-identical duplication only; visually near-duplicate images and pretraining overlap remain unmeasured.",'',
    '## Training and retained evidence','',
    'Linear probes select learning rate (1e-4/3e-4/1e-3), weight decay (1e-4/1e-2) and stopping epoch using validation macro-F1. All 16 selected checkpoints and their train/validation/test logits, labels and predictions are saved locally with hashes. Group macro-F1 uses full-test false positives; group recall is a class-macro average.','',
    f"- KNN test accuracy: {tr['knn']['accuracy']:.6f}; macro-F1: {tr['knn']['macro_f1']:.6f}.",
    f"- Balanced probe test accuracy: {tr['balanced_probe']['test']['accuracy']:.6f}; macro-F1: {tr['balanced_probe']['test']['macro_f1']:.6f}."]
    bp=tr['balanced_probe']['predictions']
    lines.append(f"- Balanced probe train/validation/test accuracy: {bp['train']['metrics']['accuracy']:.6f} / {bp['validation']['metrics']['accuracy']:.6f} / {bp['test']['metrics']['accuracy']:.6f}; train−test gap: {bp['train']['metrics']['accuracy']-bp['test']['metrics']['accuracy']:.6f}.")
    lines+=['','### Controlled long tail: five seeds, mean ± sample standard deviation','']
    for m in methods:
        values=tr['long_tail_summary'][m]['metrics']
        lines.append('- '+m+': '+ '; '.join(f"{name} {values[name]['mean']:.6f} ± {values[name]['std']:.6f}" for name in ('accuracy','macro_f1','tail_macro_f1','tail_recall'))+'.')
    lines+=['','Exact 50:1 remains blocked: 20 unique training candidates cannot realize positive integer 50:1 sampling. No replacement or evaluation data borrowing is used.','',
    '## Calibration and near-OOD','',
    f"Temperature fitted on validation only: {osr['calibration']['temperature']:.6f}. Test ECE {osr['calibration']['test']['ece_before']:.6f} → {osr['calibration']['test']['ece_after']:.6f}.",'']
    for m,r in osr['methods'].items():
        v=r['test'];lines.append(f"- {m}: AUROC {v['auroc_ood']:.6f}; AUPR-OOD {v['aupr_ood']:.6f}; FPR95-OOD {v['fpr_at_95_tpr_ood']:.6f}; known coverage {v['known_coverage']:.6f}; unknown rejection {v['unknown_rejection_rate']:.6f}.")
    lines+=['',f'Highest observed held-out AUROC: {best}. This is descriptive comparison, not test-driven model selection. No strong open-set detector or far-OOD claim is made.','',
    '## Monitoring','',
    'A fixed 2,000-image validation reference is compared with 1,000-image windows throughout. Thresholds use 200 repeated windows from the disjoint remaining validation pool; all comparisons use fixed reference-derived bandwidth. Repeated calibration windows overlap, so no independent 1% false-alarm guarantee is claimed.','']
    for split,r in mon['evaluations'].items():lines.append(f"- {split}: {len(r['windows'])} nonoverlapping windows; alert rates {json.dumps(r['alert_rates'])}; {r['unused_images']} remainder images not evaluated.")
    lines+=['','## Provenance and reproduction boundary','',
    '2.1 subsets individually hash-verified V2 frozen features by image ID; it does not re-encode images. The original encoder weight/processor revision was not recorded in 2.0 and cannot be retroactively certified. Reproducibility is therefore conditional on the hash-locked V2 feature artifacts, not a promise of raw-to-feature bitwise regeneration. Config/split locks were written before retraining. Training-critical source hashes and the Python package inventory are saved in results_v21/provenance.json.','',
    'Verification independently checks saved models against test logits, recomputes all 16 models’ classification/group metrics, checks five-seed aggregation, refits temperature, recomputes rejection thresholds/metrics, and recomputes monitoring from embeddings. The exact-search tie regression now returns the same globally lowest tied IDs for multiple shard sizes. Before full verification on a fresh workspace, run `python -m checks.v21_exact_audit` for the SIFT correctness gate (requires original SIFT files).','',
    'The release allowlist excludes images, source paths, feature arrays, checkpoints, per-example predictions, conversation transcripts and nested archives. These private artifacts stay local; public manifests retain image IDs/content hashes and permitted labels. A data-free package verifier checks every archive entry and public lock.','',
    '## Commands','', '```powershell','python -m pipelines.v21_prepare','python -m pipelines.v21_train','python -m pipelines.v21_open_set','python -m pipelines.v21_monitor','python -m checks.v21_contracts','python -m checks.v21_verify','python -m reporting.v21_results','python -m tools.package_v21','python -m checks.v21_release','```','',
    'Run from the repository root with the original local dataset/feature environment. Existing completed stage artifacts refuse overwrite. Restore historical 2.0 code from its ZIP to rerun the original implementation; current shared metric code includes the 2.1 corrections.']
    (ROOT/'docs/report/V21-RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'accuracy':summary['balanced']['probe']['accuracy'],'best_ood':best},indent=2))


if __name__=='__main__':main()
