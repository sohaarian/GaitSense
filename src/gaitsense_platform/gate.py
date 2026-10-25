"""Verification gate: Capability 2 refuses to produce clinical results without a verified, real dataset.

Rules
1. Every manifest field must be filled; placeholder text is rejected.
2. `is_synthetic=True` data are allowed only when the caller passes `allow_synthetic=True` (tests and pipeline
   checks). Every output produced from synthetic data carries `SYNTHETIC_BANNER`.
3. Each listed data file must exist and match its recorded SHA-256, so the analysed files are the verified files.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .contracts import DatasetManifest

SYNTHETIC_BANNER = "SYNTHETIC DATA - PIPELINE CHECK ONLY - NOT A CLINICAL RESULT"
PLACEHOLDERS = {"", "tbd", "todo", "unknown", "n/a", "na", "none", "?", "pending"}


class GateError(PermissionError):
    """The dataset has not been verified for a real Capability 2 analysis."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(manifest: DatasetManifest, root=".", allow_synthetic: bool = False) -> dict:
    """Return a gate record ({'status', 'banner', ...}) or raise GateError."""
    if manifest.is_synthetic:
        if not allow_synthetic:
            raise GateError("synthetic data cannot be used for a clinical analysis; pass allow_synthetic=True only "
                            "for tests and pipeline checks")
        return {"status": "synthetic", "banner": SYNTHETIC_BANNER, "dataset": manifest.name}
    empty = [k for k, v in asdict(manifest).items()
             if k not in ("notes", "file_sha256", "is_synthetic") and str(v).strip().lower() in PLACEHOLDERS]
    if empty:
        raise GateError(f"manifest incomplete: {empty}")
    if manifest.access_type not in ("open", "controlled"):
        raise GateError("access_type must be 'open' or 'controlled'")
    if not manifest.file_sha256:
        raise GateError("manifest lists no data files with checksums")
    for rel, want in manifest.file_sha256.items():
        p = Path(root) / rel
        if not p.exists():
            raise GateError(f"listed data file not found: {rel}")
        if sha256(p) != want:
            raise GateError(f"checksum mismatch for {rel}: the file differs from the verified copy")
    return {"status": "verified", "banner": "", "dataset": manifest.name, "files": sorted(manifest.file_sha256)}
