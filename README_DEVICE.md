# GaitSense device feasibility package (v0.3)

**Status: concept stage. No physical sensor has been built or measured.** `bench_analysis.py` is tested only on
simulated data with known ground truth. Passing tests show the arithmetic is correct, not that a device works,
is novel, or is clinically useful. The existing GaitRec research in this repository is unchanged.

## Engineering question
Can a wearable two-channel (vertical Fz, fore-aft Fx) force sensor tell a real change in push-off force from sensor
drift, remounting, temperature and footwear effects, better than existing multi-axis insoles?

## What `bench_analysis.py` computes
| Function | Purpose |
|---|---|
| `fit_calibration` | 2x2 sensitivity matrix + offsets by least squares; reports cross-axis coupling (%). Refuses collinear loading. |
| `validate` | Independent error (RMSE, max, bias; N and % full scale) on loads *not* used for fitting. |
| `naive_validate` | Same data, diagonal-only calibration, to show what ignoring cross-axis coupling costs. |
| `hysteresis_pct_fs` | Max up/down branch gap at matched loads, % full scale. |
| `drift_rate` | Slope of estimated force under a held load, with 95% CI. |
| `repeatability` | Within-remount SD, between-remount SD, ICC, SEM, MDC95 across occasions. |
| `bland_altman` | Bias and limits of agreement versus a reference instrument. |
| `reference_check` | Visit-time check of a known built-in test load against the stored calibration. |

```
python bench_analysis.py calibrate train.csv test.csv --fs-x 400 --fs-z 1500 --out result.json
python bench_analysis.py repeat remount.csv
pytest tests_device
```
CSV columns: see the module docstring (`ref_fx, ref_fz, raw_x, raw_z` required; `branch, time_h, mount, session, temp_c` optional).

## Bench protocol the code is built for
1. **Combined-load calibration grid** (Fx and Fz varied independently, 30+ points) on a calibrated reference rig. Fit on one set, validate on a separate randomized set.
2. **Loading/unloading cycles** for hysteresis.
3. **Constant-load hold, 8 h+**, for drift, repeated at two temperatures.
4. **Blinded remount series:** 30+ remounts, 5+ readings each, same applied load, for the repeatability/MDC figures.
5. **Reference-check tolerance** (`tol_pct`) must be set from step 4, not guessed.

## Milestones (acceptance criteria are to be set in advance, before any physical data)
- M1 sensing element, placement and mechanical structure specified; drawings and parts list.
- M2 bench calibration and verification on reference equipment.
- M3 agreement with force plate and with an existing commercial insole.
- M4 ethics-approved feasibility study in rehabilitation patients.

## Caveats carried over from the review
- Multi-axis shear insoles are published prior art; novelty, if any, would have to be in the calibration mechanism or
  mechanical coupling, and needs a dedicated patent search. Consider this before any public disclosure of design details.
- Detecting a force change is not the same as detecting recovery or work readiness.

## Software layer (`gaitsense/`, roadmap steps 16-24)
Simulation-tested only. Pipeline: `acquisition` (shared-clock Recording; CSV reader and simulator; hardware drivers
would implement `Source.read()`) -> calibration (`bench_analysis.Calibration`) -> `quality` gate (time, gaps, saturation
or clipping, flatline, too few steps) -> `metrics` (stance segmentation, braking/push-off impulse in BW*s) ->
`longitudinal` (visit summary, visit-to-visit change with MDC95, bilateral ratio) -> `audit` (hash-chained log).
Recordings that fail the quality gate return no metrics.

Design choices worth reviewing:
- `longitudinal.change` requires a *measured* remount SD (from the bench remount study). A test shows that without it,
  a 6% remount sensitivity error produces a spurious "significant change".
- The bilateral ratio is context only; the unaffected limb also changes between visits in the GaitRec data.
- The audit log is tamper-evident, not tamper-proof, and is not a substitute for a quality-system record.
- Quality thresholds (clipping run length, gap length, peak CV) are placeholders to be set from real sensor data.

### v0.3 additions
| Module | What it does | Main limits |
|---|---|---|
| `drift.py` | Fits sensitivity drift vs temperature and time from **known reference-load checks only** (never from gait), applies a term only if its CI excludes zero, warns on extrapolation or missing temperature. A test shows a genuine +15% push-off change survives correction while a ~5% drift bias is removed. | Assumes drift is linear in time and temperature. Offset drift is not modelled. Needs 6+ checks spread over both time and temperature. |
| `dashboard.py` | Self-contained HTML (inline SVG, no scripts): impulse trends with 95% intervals including remount error, change vs first usable visit against the measurement-error limit, failed visits shown as failures, warnings. | Static page; no interactivity; not usability-tested with clinicians. |
| `security.py` | PBKDF2 passwords, lockout, expiring sessions, role permissions, AES-256-GCM records bound to patient id, pseudonymised ids in filenames and logs, access/denial logging. | No security review; in-memory sessions; no network layer, MFA, key rotation or key storage; not a compliance claim. Do not store real patient data. |
| `drivers.py`, `docs/driver_interface.md` | Abstract `SensorDriver`; 18-byte framed protocol with CRC-16, sequence and timestamp wrap handling, resync, loss counting. | Protocol is a proposal; never run against hardware. |
| `tests_device/test_verification.py` | Corrupted/empty/NaN data, dead and stuck channels, calibration and cross-axis changes, and a simulation of false-alarm rate (about 5% with a measured remount SD vs more than 30% without) and detection power. | All simulated. |

Not yet done: offset drift, access control on the dashboard file itself, real-time streaming, hardware validation of any kind.
