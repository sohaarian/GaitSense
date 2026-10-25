"""Run Capability 2 on a VERIFIED linked dataset.

  python scripts/v3/run_capability2.py manifest.json visits.csv covariates.csv outcomes.csv config.json

The verification gate refuses synthetic data and any manifest that is incomplete or whose file checksums do not
match. Writes results/v3/capability2_<dataset>.json. No such dataset is available yet (docs/v3_dataset_assessment.md).
"""
import json, re, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_platform import pipeline
from gaitsense_platform.contracts import DatasetManifest

man = DatasetManifest.from_dict(json.loads(Path(sys.argv[1]).read_text()))
cfg = json.loads(Path(sys.argv[5]).read_text()); cfg["clinical_cols"] = tuple(cfg["clinical_cols"])
out = pipeline.run(man, pd.read_csv(sys.argv[2]), pd.read_csv(sys.argv[3]), pd.read_csv(sys.argv[4]),
                   pipeline.Config(**cfg), root=Path(sys.argv[1]).parent, allow_synthetic=False)
dest = ROOT / "results" / "v3" / f"capability2_{re.sub(r'[^A-Za-z0-9]+', '_', man.name)}.json"
dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(out, indent=2))
print("wrote", dest)
