"""Record that V2 artifacts keep the content they had at the original release baseline."""
import json
import subprocess
from pipelines.prepare_data import ROOT
from src.paths import resolve
from src.training.provenance import sha256,write_json

BASELINE='2fa9424a2b1ccda356028f2d3caa6d99d5814a89'
# The trees the layout migration moved, named by their baseline and current roots.
BASELINE_TREES=('releases/1.0','releases/2.0','data_v2','results_v2','configs/v2')
CURRENT_TREES=('releases/1.0','releases/2.0','data/v2.0','results/v2.0','configs/v2.0')


def main():
    # A path-level diff against the baseline cannot show content identity: the current paths do not
    # exist at the baseline, so every artifact reads as an addition, and rename detection would
    # report a content-changing move as a rename. Compare the recorded bytes of every baseline
    # artifact with the bytes on disk at its migrated path instead.
    #
    # The comparison drops CRLF before comparing. The baseline commit predates ``.gitattributes``
    # (``* -text``), so it stored LF blobs while the checked-out tree holds CRLF for the same
    # logical content; the difference is line-ending representation, not artifact content. A
    # difference that survives that normalisation is a real content change and fails the check.
    listing=subprocess.check_output(['git','ls-tree','-r',BASELINE,'--',*BASELINE_TREES],cwd=ROOT,text=True)
    before={entry.split('\t',1)[1]:entry.split('\t',1)[0].split()[2] for entry in listing.splitlines() if entry}
    mapped={path:resolve(path).relative_to(ROOT).as_posix() for path in before}
    missing=sorted(current for current in mapped.values() if not (ROOT/current).is_file())
    assert not missing,missing
    identical=[]
    representation=[]
    changed=[]
    for path,current in sorted(mapped.items()):
        original=subprocess.run(['git','cat-file','blob',f'{BASELINE}:{path}'],cwd=ROOT,capture_output=True,check=True).stdout
        on_disk=(ROOT/current).read_bytes()
        if original==on_disk:
            identical.append(current)
        elif original.replace(b'\r\n',b'\n')==on_disk.replace(b'\r\n',b'\n'):
            representation.append(current)
        else:
            changed.append(current)
    assert not changed,changed
    tracked=subprocess.check_output(['git','ls-files','-z','--',*CURRENT_TREES],cwd=ROOT,text=True)
    extra=sorted({path for path in tracked.split('\0') if path}-set(mapped.values()))
    checks={}
    for version,name in [('1.0','Assignment1-submission.zip'),('2.0','Assignment1-2.0.zip')]:
        path=ROOT/'releases'/version/name
        actual=sha256(path);expected=path.with_suffix('.zip.sha256').read_text().split()[0]
        assert actual==expected
        checks[version]={'sha256':actual,'bytes':path.stat().st_size}
    write_json(ROOT/'results/v2.1/checks/historical_integrity.json',{'status':'passed','baseline_commit':BASELINE,'method':'every artifact tracked at the baseline commit under releases/1.0, releases/2.0, data_v2, results_v2 and configs/v2 was mapped to its current path and its recorded baseline bytes compared with the bytes on disk; a difference that survives CRLF-to-LF normalisation is a content change and fails the check','historical_paths_compared':len(mapped),'historical_paths_missing':missing,'content_changed_paths':changed,'representation_only_paths':representation,'paths_identical_at_byte_level':identical,'paths_added_after_baseline':extra,'archives':checks,'note':'New audit evidence under results/v2.1/checks is supplemental; original tracked V2 artifacts are unchanged in content; historical paths were renamed to the versioned layout without content changes'})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
