"""Record that V2 artifacts remain byte-identical to the original release baseline."""
import json
import subprocess
from pipelines.prepare_data import ROOT
from src.training.provenance import sha256,write_json


def main():
    baseline='2fa9424a2b1ccda356028f2d3caa6d99d5814a89'
    changed=subprocess.check_output(['git','diff','--name-only',baseline,'--','releases/1.0','releases/2.0','data_v2','results_v2','configs/v2'],cwd=ROOT,text=True).splitlines()
    assert not changed,changed
    checks={}
    for version,name in [('1.0','Assignment1-submission.zip'),('2.0','Assignment1-2.0.zip')]:
        path=ROOT/'releases'/version/name
        actual=sha256(path);expected=path.with_suffix('.zip.sha256').read_text().split()[0]
        assert actual==expected
        checks[version]={'sha256':actual,'bytes':path.stat().st_size}
    write_json(ROOT/'results_v21/checks/historical_integrity.json',{'status':'passed','baseline_commit':baseline,'tracked_historical_paths_changed':changed,'archives':checks,'note':'New audit evidence under results_v2/checks is supplemental; original tracked V2 artifacts are unchanged'})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
