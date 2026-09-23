# Post-release CI correction

The immutable 2.1 experimental archive was built from commit `13e7a16873231ac961958d9a3db082a4d8715a00` and has SHA-256 `425ae9c9f3871e372d0f2cf79f41d12ef5b2eabc6463e4fbb32dcef6c4ad77db`.

The first clean GitHub runner exposed an unnecessary dataset-preparation import in the legacy CPU contracts. Commit `f230558` replaces `from pipelines.prepare_data import ROOT` in `checks/v2_ci.py` with `ROOT = Path(__file__).resolve().parents[1]` (Path is already imported). This changes no experiment implementation or results.

Both the original 14 CPU contracts and the revision regression contracts pass on the clean Windows runner at:
https://github.com/GoingFall/stat6207-assignment1-seeing-similarity/actions/runs/35806714906

Use main at `f230558` or later for the minimal-dependency CI checks. The attached `v2_ci.py` asset is the corrected standalone file, to replace `checks/v2_ci.py` when running checks from the historical 2.1 ZIP in a minimal environment. The original ZIP/tag remains unmodified; this note and file are supplemental release assets.
