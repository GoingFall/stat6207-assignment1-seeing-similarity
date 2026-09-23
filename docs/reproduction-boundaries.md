# Reproduction boundaries

## Full Version 2 replay

Full replay is available only on a machine that has the private source manifests/images and the hash-locked V2 embeddings. The repository deliberately does not redistribute those restricted assets. Run:

```powershell
conda run -n stat6207-a1 python -m checks.local_replay_preflight
```

`ready` means the required local inputs were found and the configuration hash matched the lock. `local-only-verification` means the machine can verify committed locks and retained outputs but cannot honestly rerun the complete pipeline. `verification-failed` means the inputs were present but a recorded hash did not match, which is a real regression rather than a missing-data boundary. A lock that omits the config digest is treated as a failure, not as a reason to skip the check. The command never downloads or fabricates missing data.

## Version 1.0 bulk replay

The large Version 1.0 inputs, embeddings and generated media remain gitignored because of size and rights constraints. The tracked protocol, manifests, provenance probes, checksums and JSON evidence define the replay contract. A complete 1.0 replay requires restoring the local `data/v1.0/` bulk inputs and `results/v1.0/` prediction/embedding artifacts from the authorized local backup, then running the existing 1.0 preparation, evaluation and verification entrypoints.

The absence of bulk files from Git is intentional and is not treated as a successful replay. `checks.local_replay_preflight` and the existing verification scripts must pass on the data-bearing machine before claiming full reproducibility.

## Tooling

Ruff is pinned in `requirements.txt` and installed in the `stat6207-a1` environment. Repository Quality Guard is installed globally at `~/.agents/skills/repository-quality-guard`; this workstation's `.git/hooks/pre-push` invokes that global runtime. Because Git hooks are local metadata and the upstream RQG release manifest currently rejects deployment of its checked-out runtime, this hook is a workstation safeguard rather than a clone-reproducible project asset. A fresh clone must install a verified project-local RQG runtime before enabling an equivalent hook.
