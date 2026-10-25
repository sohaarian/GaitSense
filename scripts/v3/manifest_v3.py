"""v3.0 manifest: platform code, v3 outputs and the frozen v2.0 experiment they build on."""
import hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn
ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_platform import __version__
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
code = sorted(list((ROOT / "src" / "gaitsense_platform").glob("*.py")) + list((ROOT / "scripts" / "v3").glob("*.py")))
v2 = json.loads((ROOT / "results" / "ai" / "run_manifest.json").read_text())
man = {"platform_version": __version__, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "builds_on_v2_experiment": v2["experiment_id"],
       "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
       "capability2_status": "no verified linked dataset; synthetic pipeline checks only",
       "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
       "outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "results" / "v3").glob("*.json")) if p.name != "manifest_v3.json"}}
(ROOT / "results" / "v3" / "manifest_v3.json").write_text(json.dumps(man, indent=2))
print("v3 manifest:", len(man["outputs_sha256"]), "outputs; builds on", man["builds_on_v2_experiment"])
