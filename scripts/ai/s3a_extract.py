"""Stage 3a: extract trial-level features and normalised curves from the GaitRec RAW files.

Usage: python scripts/ai/s3a_extract.py [RAW_DIR]   (default data/raw)
Writes data/derived/ai_trials.csv.gz, data/derived/ai_curves_{left,right}.npz, results/ai/provenance_raw.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import ingest

raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
meta = ingest.load_metadata(ROOT / "data" / "GRF_metadata.csv")
bw = meta.set_index("SESSION_ID").BODY_WEIGHT
out = ROOT / "results" / "ai"; out.mkdir(parents=True, exist_ok=True)
prov = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
for f in sorted(raw.glob("GRF_F_*_RAW_*.csv")) + [ROOT / "data" / "GRF_metadata.csv"]:
    prov["files"][f.name] = {"bytes": f.stat().st_size, "sha256": ingest.sha256(f)}
tables = []
for side in ("left", "right"):
    t, c = ingest.extract_side(raw, side, bw)
    np.savez_compressed(ROOT / "data" / "derived" / f"ai_curves_{side}.npz", **c)
    tables.append(t); print(side, len(t), "trials,", int(t.valid.sum()), "valid", flush=True)
T = pd.concat(tables, ignore_index=True)
T.to_csv(ROOT / "data" / "derived" / "ai_trials.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(out / "provenance_raw.json").write_text(json.dumps(prov, indent=2))
print("done", len(T))
