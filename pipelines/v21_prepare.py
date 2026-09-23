"""Freeze content-cleaned V2.1 manifests and subset hash-locked V2 features."""
import json
from collections import defaultdict, Counter
from pathlib import Path
import platform
import subprocess
import numpy as np
from pipelines.prepare_data import ROOT
from src.training.provenance import sha256, write_json, artifact, assert_content_disjoint
from src.training.long_tail import sample_ids
from src.paths import resolve


def main():
    output = ROOT / 'data/v2.1'
    lock_path = output / 'manifests/lock.json'
    if lock_path.exists():
        raise RuntimeError('V2.1 lock exists; refusing to rewrite')
    config_path = ROOT / 'configs/v2.1/experiment.json'
    config = json.loads(config_path.read_text())
    old = json.loads((ROOT / 'results/v2.0/inat/encoding.json').read_text())
    source_path = ROOT / 'data/v2.0/manifests/inat_birds_images.jsonl'
    image_lock = json.loads((ROOT / 'data/v2.0/manifests/inat_birds_images_lock.json').read_text())
    assert sha256(source_path) == image_lock['manifest_sha256']
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        assert sha256(resolve(row['local_path'])) == row['sha256']
        groups[row['sha256']].append(row)
    priority = {s: i for i, s in enumerate(('train', 'validation', 'test', 'unknown_development', 'unknown_test'))}
    clean, removed = [], []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (priority[r['split']], r['image_id']))
        conflict = len({r['class_id'] for r in group}) > 1
        keep = [] if conflict else ordered[:1]
        clean.extend(keep)
        for row in (ordered if conflict else ordered[1:]):
            removed.append({k: row[k] for k in ('image_id', 'class_id', 'split', 'sha256')} | {'reason': 'conflicting_labels' if conflict else 'duplicate_content'})
    buckets = defaultdict(list)
    for row in clean:
        buckets[(row['split'], row['class_id'])].append(row)
    selected, unused = [], []
    for (split, cls), values in sorted(buckets.items()):
        values.sort(key=lambda r: r['image_id'])
        count = config['split']['known_per_class'].get(split, len(values))
        if len(values) < count:
            raise AssertionError(f'Insufficient clean candidates: {split}/{cls}')
        selected.extend(values[:count])
        unused.extend(r['image_id'] for r in values[count:])
    assert_content_disjoint(selected)
    manifests = {}
    for split in priority:
        public = [{k: r[k] for k in ('image_id', 'class_id', 'split', 'sha256', 'bytes', 'license')} for r in selected if r['split'] == split]
        public.sort(key=lambda r: r['image_id'])
        path = output / f'manifests/{split}.json'
        write_json(path, public)
        manifests[split] = artifact(path, ROOT) | {'images': len(public), 'classes': len({r['class_id'] for r in public})}
    known = [{r['class_id'] for r in selected if r['split'] == s} for s in ('train', 'validation', 'test')]
    assert known[0] == known[1] == known[2] and len(known[0]) == 1000
    ud = {r['class_id'] for r in selected if r['split'] == 'unknown_development'}
    ut = {r['class_id'] for r in selected if r['split'] == 'unknown_test'}
    assert not (known[0] & ud or known[0] & ut or ud & ut)
    candidates = defaultdict(list)
    for row in selected:
        if row['split'] == 'train': candidates[row['class_id']].append(row['image_id'])
    runs, blocked = [], []
    for seed in config['long_tail']['seeds']:
        run = sample_ids(candidates, 10, seed)
        path = output / f'manifests/long_tail/ratio-10-seed-{seed}.json'
        write_json(path, run)
        runs.append(artifact(path, ROOT) | {'seed': seed, 'ratio': 10})
        try: sample_ids(candidates, 50, seed)
        except ValueError as e: blocked.append({'seed': seed, 'ratio': 50, 'reason': str(e)})
        else: raise AssertionError('50:1 must be blocked')
    frozen = {'config': config, 'config_sha256': sha256(config_path), 'source_manifest_sha256': sha256(source_path), 'source_encoding': artifact(ROOT/'results/v2.0/inat/encoding.json', ROOT), 'manifests': manifests, 'removed': removed, 'unused_known_ids': unused, 'long_tail': {'runs': runs, 'blocked': blocked}}
    # Publish split/config lock before constructing any revised training artifacts.
    write_json(lock_path, frozen)
    result = {'status': 'running', 'lock': artifact(lock_path, ROOT), 'model': 'V2 hash-locked frozen DINOv2 features', 'splits': {}}
    for split, record in manifests.items():
        source = old['splits'][split]['artifacts']
        for a in source.values(): assert sha256(resolve(a['path'])) == a['sha256']
        ids = np.load(resolve(source['image_ids']['path']))
        position = {int(i): j for j, i in enumerate(ids)}
        public = json.loads((resolve(record['path'])).read_text())
        positions = [position[r['image_id']] for r in public]
        artifacts = {}
        for kind in ('embeddings', 'labels', 'image_ids'):
            values = np.load(resolve(source[kind]['path']), mmap_mode='r')[positions]
            path = output/f'embeddings/inat_birds/{split}_{kind}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values, allow_pickle=False)
            artifacts[kind] = artifact(path, ROOT)
        result['splits'][split] = {'artifacts': artifacts, 'source_artifacts': source, 'images': len(public)}
    result['status'] = 'passed'
    write_json(ROOT/'results/v2.1/inat/encoding.json', result)
    provenance = {'python': platform.python_version(), 'platform': platform.platform(), 'pip_freeze': subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze'], text=True).splitlines(), 'code': {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ('src','pipelines','checks','tools') for p in (ROOT/folder).rglob('*.py')}, 'frozen_releases': {v: artifact(ROOT/p, ROOT) for v,p in [('1.0','releases/1.0/Assignment1-submission.zip'),('2.0','releases/2.0/Assignment1-2.0.zip')]}}
    # Keep the audited layout-migration annex: it records the pre/post hash of every file the directory
# normalisation touched. If the sources changed again the annex hashes no longer match and
# checks.v21_verify fails loudly instead of silently accepting a stale audit.
provenance_path=ROOT/'results/v2.1/provenance.json'
if provenance_path.is_file():
    annex=json.loads(provenance_path.read_text(encoding='utf-8')).get('layout_migration')
    if annex:provenance['layout_migration']=annex
write_json(provenance_path, provenance)
    print(json.dumps({'splits': manifests, 'quarantined_or_deduplicated': len(removed), 'blocked_50': len(blocked)}, indent=2))


if __name__ == '__main__': main()
