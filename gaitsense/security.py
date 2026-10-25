"""Authentication, role-based access, encryption at rest and access logging.

HONEST SCOPE: a research prototype using standard primitives (PBKDF2-HMAC-SHA256, scrypt, AES-256-GCM via the
`cryptography` package). It has NOT had a security review or penetration test, keeps sessions in process memory, has no
network layer, no MFA, no key rotation or hardware key storage, and is not a claim of HIPAA / GDPR / IEC 62304 / FDA
compliance. Do not store real patient data with it.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROLES: Dict[str, frozenset] = {
    "admin": frozenset({"manage_users", "read_audit"}),
    "clinician": frozenset({"read_record", "write_record"}),
    "technician": frozenset({"write_record"}),
    "auditor": frozenset({"read_audit"}),
}
MIN_PASSWORD_LEN = 12


class AuthError(Exception):
    pass


class PermissionDenied(Exception):
    pass


def _pbkdf2(pw: str, salt: bytes, iters: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iters)


def derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode(), salt=salt, n=2 ** 15, r=8, p=1, dklen=32, maxmem=2 ** 26)


def _pid_hash(pid: str, pepper: bytes) -> str:
    """Pseudonymous id for logs and filenames: patient identifiers never appear in either."""
    return hmac.new(pepper, pid.encode(), hashlib.sha256).hexdigest()[:20]


class SecureService:
    def __init__(self, root: str, audit_log, key: bytes, pbkdf2_iters: int = 600_000,
                 max_failures: int = 5, lockout_s: int = 900, session_ttl_s: int = 1800,
                 clock: Callable[[], float] = time.time):
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        self.root = Path(root); (self.root / "records").mkdir(parents=True, exist_ok=True)
        self.users_path = self.root / "users.json"
        self.audit, self._aead = audit_log, AESGCM(key)
        self._pepper = hashlib.sha256(b"pid-pepper" + key).digest()
        self.iters, self.max_failures, self.lockout_s, self.ttl, self.clock = pbkdf2_iters, max_failures, lockout_s, session_ttl_s, clock
        self._sessions: Dict[str, dict] = {}

    # ---------------- users
    def _load(self) -> dict:
        return json.loads(self.users_path.read_text()) if self.users_path.exists() else {}

    def _save(self, users: dict):
        tmp = self.users_path.with_suffix(".tmp"); tmp.write_text(json.dumps(users)); os.replace(tmp, self.users_path)

    def create_user(self, actor_token: Optional[str], name: str, password: str, role: str):
        """The first user must be an admin and is created without a token (bootstrap); afterwards only admins can."""
        users = self._load()
        if users:
            self.authorize(actor_token, "manage_users")
        elif role != "admin":
            raise ValueError("first user must be an admin")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if len(password) < MIN_PASSWORD_LEN:
            raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        if name in users:
            raise ValueError("user exists")
        salt = secrets.token_bytes(16)
        users[name] = {"salt": salt.hex(), "hash": _pbkdf2(password, salt, self.iters).hex(), "iters": self.iters,
                       "role": role, "failures": 0, "locked_until": 0.0}
        self._save(users)
        self.audit.append("user_created", user=name, role=role)

    # ---------------- login / sessions
    def login(self, name: str, password: str) -> str:
        users = self._load(); u = users.get(name); now = self.clock()
        if u and u["locked_until"] > now:
            self.audit.append("login_locked", user=name); raise AuthError("account temporarily locked")
        # always do the hash so unknown users and wrong passwords cost the same
        salt = bytes.fromhex(u["salt"]) if u else b"\0" * 16
        got = _pbkdf2(password, salt, u["iters"] if u else self.iters)
        ok = bool(u) and hmac.compare_digest(got, bytes.fromhex(u["hash"]))
        if not ok:
            if u:
                u["failures"] += 1
                if u["failures"] >= self.max_failures:
                    u["locked_until"] = now + self.lockout_s; u["failures"] = 0
                    self.audit.append("lockout", user=name)
                self._save(users)
            self.audit.append("login_failed", user=name)
            raise AuthError("invalid credentials")
        u["failures"] = 0; self._save(users)
        tok = secrets.token_urlsafe(32)
        self._sessions[tok] = {"user": name, "role": u["role"], "exp": now + self.ttl}
        self.audit.append("login_ok", user=name)
        return tok

    def logout(self, token: str):
        s = self._sessions.pop(token, None)
        if s:
            self.audit.append("logout", user=s["user"])

    def authorize(self, token: Optional[str], permission: str) -> str:
        s = self._sessions.get(token or "")
        if not s or s["exp"] < self.clock():
            self._sessions.pop(token or "", None)
            self.audit.append("access_denied", reason="no_valid_session", permission=permission)
            raise PermissionDenied("not logged in or session expired")
        if permission not in ROLES[s["role"]]:
            self.audit.append("access_denied", user=s["user"], role=s["role"], permission=permission)
            raise PermissionDenied(f"role {s['role']!r} lacks {permission!r}")
        return s["user"]

    # ---------------- encrypted records
    def _path(self, pid: str) -> Path:
        return self.root / "records" / (_pid_hash(pid, self._pepper) + ".bin")

    def put_record(self, token: str, patient_id: str, data: dict):
        user = self.authorize(token, "write_record")
        nonce = secrets.token_bytes(12)
        aad = _pid_hash(patient_id, self._pepper).encode()       # binds ciphertext to its patient
        blob = nonce + self._aead.encrypt(nonce, json.dumps(data, sort_keys=True).encode(), aad)
        self._path(patient_id).write_bytes(blob)
        self.audit.append("record_write", user=user, pid=_pid_hash(patient_id, self._pepper))

    def get_record(self, token: str, patient_id: str) -> dict:
        user = self.authorize(token, "read_record")
        p = self._path(patient_id); pid = _pid_hash(patient_id, self._pepper)
        if not p.exists():
            self.audit.append("record_read_missing", user=user, pid=pid); raise KeyError("no such record")
        blob = p.read_bytes()
        try:
            out = json.loads(self._aead.decrypt(blob[:12], blob[12:], pid.encode()))
        except InvalidTag:
            self.audit.append("record_integrity_failure", user=user, pid=pid)
            raise ValueError("record failed integrity check") from None
        self.audit.append("record_read", user=user, pid=pid)
        return out

    def read_audit(self, token: str) -> str:
        self.authorize(token, "read_audit")
        return Path(self.audit.path).read_text()
