import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaitsense import audit as au
from gaitsense.security import AuthError, PermissionDenied, SecureService

PW = "correct horse battery"


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


@pytest.fixture
def svc(tmp_path):
    clk = Clock()
    s = SecureService(str(tmp_path / "s"), au.AuditLog(str(tmp_path / "audit.log")), key=bytes(range(32)),
                      pbkdf2_iters=1000, clock=clk)
    s.clk = clk; s.tmp = tmp_path
    s.create_user(None, "root", PW, "admin")
    return s


def mk(svc, name, role):
    svc.create_user(svc.login("root", PW), name, PW + "x", role)
    return svc.login(name, PW + "x")


def test_bootstrap_rules_and_password_policy(tmp_path):
    s = SecureService(str(tmp_path / "t"), au.AuditLog(str(tmp_path / "a")), key=bytes(32), pbkdf2_iters=1000)
    with pytest.raises(ValueError):
        s.create_user(None, "c", PW, "clinician")           # first user must be admin
    with pytest.raises(ValueError):
        s.create_user(None, "root", "short", "admin")
    s.create_user(None, "root", PW, "admin")
    with pytest.raises(PermissionDenied):
        s.create_user(None, "evil", PW, "admin")             # no token after bootstrap


def test_non_admin_cannot_create_users(svc):
    clin = mk(svc, "dr", "clinician")
    with pytest.raises(PermissionDenied):
        svc.create_user(clin, "x", PW + "y", "admin")


def test_wrong_password_unknown_user_and_lockout_then_recovery(svc):
    for _ in range(5):
        with pytest.raises(AuthError):
            svc.login("root", "wrong password!!")
    with pytest.raises(AuthError, match="locked"):
        svc.login("root", PW)                                # right password still refused while locked
    svc.clk.t += 901
    assert svc.login("root", PW)
    with pytest.raises(AuthError):
        svc.login("nobody", PW)


def test_role_permissions(svc):
    clin, tech, aud = mk(svc, "dr", "clinician"), mk(svc, "tech", "technician"), mk(svc, "aud", "auditor")
    svc.put_record(tech, "P1", {"x": 1})
    with pytest.raises(PermissionDenied):
        svc.get_record(tech, "P1")                           # technician can write, not read
    assert svc.get_record(clin, "P1") == {"x": 1}
    with pytest.raises(PermissionDenied):
        svc.get_record(aud, "P1")
    assert "record_read" in svc.read_audit(aud)
    with pytest.raises(PermissionDenied):
        svc.read_audit(clin)


def test_session_expiry_and_logout(svc):
    clin = mk(svc, "dr", "clinician")
    svc.clk.t += 1801
    with pytest.raises(PermissionDenied):
        svc.get_record(clin, "P1")
    t2 = svc.login("dr", PW + "x"); svc.logout(t2)
    with pytest.raises(PermissionDenied):
        svc.get_record(t2, "P1")
    with pytest.raises(PermissionDenied):
        svc.get_record(None, "P1")


def test_encryption_at_rest_no_plaintext_and_no_identifiers_in_files_or_log(svc):
    tech = mk(svc, "tech", "technician")
    svc.put_record(tech, "Jane Q Patient 12345", {"note": "SECRET-DIAGNOSIS", "push": 0.0421})
    blobs = b"".join(p.read_bytes() for p in (svc.tmp / "s" / "records").iterdir())
    names = " ".join(p.name for p in (svc.tmp / "s" / "records").iterdir())
    assert b"SECRET-DIAGNOSIS" not in blobs and b"0.0421" not in blobs
    assert "Jane" not in names and "12345" not in names
    log = (svc.tmp / "audit.log").read_text()
    assert "Jane" not in log and "12345" not in log and PW not in log and "SECRET" not in log


def test_tampering_and_swapping_ciphertext_is_detected(svc):
    clin, tech = mk(svc, "dr", "clinician"), mk(svc, "tech", "technician")
    svc.put_record(tech, "A", {"v": 1}); svc.put_record(tech, "B", {"v": 2})
    pa, pb = svc._path("A"), svc._path("B")
    blob = bytearray(pa.read_bytes()); blob[-1] ^= 1; pa.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="integrity"):
        svc.get_record(clin, "A")
    pa.write_bytes(pb.read_bytes())                          # patient B's record presented as patient A's
    with pytest.raises(ValueError, match="integrity"):
        svc.get_record(clin, "A")
    assert "record_integrity_failure" in (svc.tmp / "audit.log").read_text()


def test_wrong_key_cannot_read(svc, tmp_path):
    tech = mk(svc, "tech", "technician"); svc.put_record(tech, "P", {"v": 1})
    other = SecureService(str(svc.tmp / "s"), au.AuditLog(str(tmp_path / "o.log")), key=bytes([1]) * 32, pbkdf2_iters=1000)
    tok = other.login("root", PW)
    other.create_user(tok, "dr", PW + "x", "clinician")
    with pytest.raises((ValueError, KeyError)):
        other.get_record(other.login("dr", PW + "x"), "P")


def test_audit_chain_still_verifies_after_all_of_it(svc):
    mk(svc, "dr", "clinician")
    with pytest.raises(AuthError):
        svc.login("dr", "nope nope nope")
    assert svc.audit.verify()
