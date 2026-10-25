"""Stage 9: reproducibility manifest - software versions, seeds, configuration, input and output checksums."""
import hashlib, json, platform, sys, time
from pathlib import Path

import numpy as np, pandas as pd, scipy, sklearn, matplotlib

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import SEED, __version__

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
A = ROOT / "results" / "ai"
code = sorted(list((ROOT / "src" / "gaitsense_ai").glob("*.py")) + list((ROOT / "scripts" / "ai").glob("*.py")))
code_hash = hashlib.sha256("".join(sha(p) for p in code).encode()).hexdigest()
sel = json.loads((A / "model_selection.json").read_text())
man = {
    "experiment_id": f"gaitsense-ai-{__version__}-{code_hash[:12]}",
    "package_version": __version__, "seed": SEED, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    "protocol_sha256": sha(ROOT / "docs" / "ai_protocol.md"),
    "protocol_frozen": (ROOT / "docs" / "ai_protocol.sha256").read_text().splitlines()[1],
    "configuration": {"partition": "60/20/20 patient-level, stratified by injury class", "bootstrap_B": 2000,
                      "matched_alert_rate": 0.10, "z": 1.959963984540054, "min_valid_trials_per_leg": 3,
                      "time_unit": "months = days / 30.44"},
    "model_choices": {t: {k: v[k] for k in ("M1_params", "M4_params", "M2_chosen", "M3_chosen", "H1_comparator")}
                      for t, v in sel["targets"].items()},
    "model_parameters": {t: v["hyperparameters"] for t, v in sel["targets"].items()},
    "timestamps": {"model_selection_finished": sel["finished"],
                   "test_opened": json.loads((A / "test_opened.json").read_text())["opened"]},
    "inputs": json.loads((A / "provenance_raw.json").read_text())["files"],
    "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
    "outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(A.glob("*")) if p.suffix in (".json", ".csv", ".gz")
                       and p.name != "run_manifest.json"},
}
(A / "run_manifest.json").write_text(json.dumps(man, indent=2, default=str))
print(man["experiment_id"], len(man["outputs_sha256"]), "outputs")
