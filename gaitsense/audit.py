"""Step 24: append-only, hash-chained audit log. Tamper-EVIDENT (edits/removals break the chain), not tamper-proof."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from . import __version__


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = Path(path)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        lines = self.path.read_text().splitlines()
        return json.loads(lines[-1])["hash"] if lines else "0" * 64

    def append(self, event: str, **details) -> dict:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "software": __version__,
               "details": details, "prev": self._last_hash()}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def verify(self) -> bool:
        prev = "0" * 64
        if not self.path.exists():
            return True
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != h:
                return False
            prev = h
        return True
