"""PyTorch version of the shear-study CNN, for running on your own computer.

STATUS: written as a learning exercise. It has NOT been run in this project, because PyTorch could not be
installed in the analysis environment. The reported CNN results come from the NumPy implementation
(src/gaitrec_rehab/cnn1d.py, scripts/shear_cnn.py), which has the same architecture and is gradient-checked.

Exercise: run this, then compare its validation loss and test predictions with data/derived/cnn_preds.npz.
They will not match exactly (different random initialisation and numerics), but the push-off change CCC
should be in the same range. If it is far off, find out why before trusting either.

Setup:  pip install torch numpy pandas scikit-learn
Run:    python scripts/shear_cnn_pytorch.py      (needs data/derived/steps_*.npz from scripts/extract_steps.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).parents[1]; sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from shear_data import load_steps, patient_split

torch.manual_seed(0)

# ---- 1. Data: same inputs, targets and patient-level split as the NumPy CNN ----
st, C, _, _ = load_steps()
keep = (st.role != "healthy").to_numpy(); st, C = st[keep].reset_index(drop=True), C[keep]
split = st.SUBJECT_ID.map(patient_split(st.SUBJECT_ID)).to_numpy()   # no patient in two sets
cov = np.column_stack([st.stance_s, st.BODY_MASS, (st.role == "affected").astype(float)] +
                      [(st.CLASS_LABEL == g).astype(float) for g in "KAHC"])
Y = st[["brake", "push"]].to_numpy()
tr, va = split == "train", split == "val"
# standardise with TRAINING statistics only (using all patients here would leak test information)
norm = lambda a: (a - a[tr].mean(0)) / a[tr].std(0)
Xc, Xv, Yz = norm(C), norm(cov), norm(Y)
T = lambda a: torch.tensor(a, dtype=torch.float32)


# ---- 2. Model: same architecture as src/gaitrec_rehab/cnn1d.py ----
class ShearCNN(nn.Module):
    def __init__(self, n_cov, segments=6):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
                                  nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(segments)   # mean within time segments: keeps WHEN things happen
        self.head = nn.Sequential(nn.Linear(32 * segments + n_cov, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, curve, cov):
        h = self.pool(self.conv(curve.unsqueeze(1))).flatten(1)
        return self.head(torch.cat([h, cov], 1))


model = ShearCNN(cov.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
xtr, vtr, ytr = T(Xc[tr]), T(Xv[tr]), T(Yz[tr])
xva, vva, yva = T(Xc[va]), T(Xv[va]), T(Yz[va])

# ---- 3. Train with early stopping on VALIDATION patients ----
best, wait, best_state = np.inf, 0, None
for epoch in range(40):
    model.train()
    for b in torch.randperm(len(ytr)).split(256):
        opt.zero_grad()
        loss = ((model(xtr[b], vtr[b]) - ytr[b]) ** 2).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = ((model(xva, vva) - yva) ** 2).mean().item()
    print(f"epoch {epoch + 1}: validation MSE {vl:.4f}")
    if vl < best - 1e-5:
        best, wait, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= 5:
            break
model.load_state_dict(best_state)

# ---- 4. Predict every step and save, in body-weight-seconds ----
with torch.no_grad():
    pred = model(T(Xc), T(Xv)).numpy() * Y[tr].std(0) + Y[tr].mean(0)
np.savez(ROOT / "data" / "derived" / "cnn_preds_pytorch.npz", pred=pred, session=st.SESSION_ID.to_numpy(),
         trial=st.TRIAL_ID.to_numpy())
print("saved; best validation MSE", round(best, 4))
