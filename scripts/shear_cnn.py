"""Post-hoc addition: a true 1D CNN (src/gaitrec_rehab/cnn1d.py, NumPy) for the shear study.

The protocol's M-cnn was replaced by an MLP because no deep-learning library was available (deviation S1).
This script adds the CNN afterwards, so it is reported as a post-hoc model. "The AI" chosen under the
protocol (M-gb) is unchanged. Same inputs as the MLP: 101-point curve, stance time, body mass, leg, injury group.
Writes data/derived/cnn_preds.npz, aligned to shear_data.load_steps() patient rows.
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parents[1]; sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from shear_data import load_steps, patient_split
from gaitrec_rehab.cnn1d import CNN1D

st, C, _, _ = load_steps()
keep = (st.role != "healthy").to_numpy(); st, C = st[keep].reset_index(drop=True), C[keep].astype(float)
sp = st.SUBJECT_ID.map(patient_split(st.SUBJECT_ID)).to_numpy()
cov = np.column_stack([st.stance_s, st.BODY_MASS, (st.role == "affected").astype(float)] +
                      [(st.CLASS_LABEL == g).astype(float) for g in "KAHC"])
tr, va = sp == "train", sp == "val"
cm, cs = C[tr].mean(0), C[tr].std(0); vm, vs = cov[tr].mean(0), cov[tr].std(0)
Xc, Xv = (C - cm) / cs, (cov - vm) / vs
Y = st[["brake", "push"]].to_numpy(); ym, ys = Y[tr].mean(0), Y[tr].std(0); Yz = (Y - ym) / ys
t0 = time.time()
net = CNN1D(n_cov=cov.shape[1], n_out=2, seed=0).fit(Xc[tr], Xv[tr], Yz[tr], val=(Xc[va], Xv[va], Yz[va]),
                                                       epochs=40, patience=5, verbose=True)
pred = net.predict(Xc, Xv) * ys + ym
np.savez(ROOT / "data" / "derived" / "cnn_preds.npz", pred=pred, session=st.SESSION_ID.to_numpy(), trial=st.TRIAL_ID.to_numpy(),
         side=st.side.to_numpy(), val_history=np.array(net.history))
print(f"done in {time.time() - t0:.0f}s, epochs {len(net.history)}, best val mse {min(net.history):.4f}")
