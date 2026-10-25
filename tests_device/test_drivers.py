import io
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analysis as ba
from gaitsense import acquisition as acq, drivers as dv, pipeline

INFO = dv.DeviceInfo("SN001", "0.1", 200.0, 1e-6, 1e-6, "R")


def rec(n=30):
    return acq.SimulatedSource(acq.SimConfig(n_steps=n)).read()


def triples(r):
    return {(round(t, 6), round(x / 1e-6), round(z / 1e-6)) for t, x, z in zip(r.t, r.raw_x, r.raw_z)}


def test_clean_roundtrip():
    r = rec(); b = dv.simulate_device_bytes(r, INFO)
    d = dv.StreamDriver(io.BytesIO(b), INFO); out = d.read()
    assert len(out.t) == len(r.t) and d.stats.crc_errors == 0 and d.stats.dropped_frames == 0
    assert np.allclose(out.raw_x, r.raw_x, atol=1e-6) and np.allclose(out.raw_z, r.raw_z, atol=1e-6)
    assert np.allclose(out.t, r.t - r.t[0], atol=1e-6) and np.all(np.diff(out.t) > 0)


def test_corrupted_bytes_are_rejected_never_accepted_as_data():
    r = rec(); b = bytearray(dv.simulate_device_bytes(r, INFO)); rng = np.random.default_rng(0)
    for i in rng.choice(len(b), size=len(b) // 200, replace=False):
        b[i] ^= 0xFF
    d = dv.StreamDriver(io.BytesIO(bytes(b)), INFO); out = d.read()
    assert d.stats.crc_errors > 0 and 0 < len(out.t) < len(r.t)
    good = {(int(round(t * 1e6)), int(round(x / 1e-6)), int(round(z / 1e-6))) for t, x, z in zip(r.t - r.t[0], r.raw_x, r.raw_z)}
    # every frame that survived is bit-identical to an original frame
    assert all((int(round(t * 1e6)), int(round(x / 1e-6)), int(round(z / 1e-6))) in good for t, x, z in zip(out.t, out.raw_x, out.raw_z))


def test_chunking_does_not_change_the_result():
    b = dv.simulate_device_bytes(rec(5), INFO)
    p1, p2 = dv.FrameParser(), dv.FrameParser()
    whole = list(p1.feed(b)); parts = []
    for i in range(len(b)):
        parts += list(p2.feed(b[i:i + 1]))
    assert whole == parts and p1.stats.frames_ok == p2.stats.frames_ok


def test_garbage_prefix_truncated_tail_and_empty_stream():
    b = dv.simulate_device_bytes(rec(5), INFO)
    out = dv.StreamDriver(io.BytesIO(b"\x00\xa5\x01garbage" + b + b[:7]), INFO).read()
    assert len(out.t) == len(b) // dv.FRAME_LEN
    e = dv.StreamDriver(io.BytesIO(b""), INFO).read()
    assert len(e.t) == 0
    res = pipeline.analyze(e, ba.Calibration([[1, 0], [0, 1]], [0, 0], 1.0), 750, 200, 300, 1200)
    assert not res["usable"] and "empty" in res["flags"]


def test_dropped_frames_counted_and_big_gap_is_flagged_by_quality_gate():
    r = rec(); frames = [dv.encode_frame(k, int(round(t * 1e6)), int(x / 1e-6), int(z / 1e-6)) for k, (t, x, z) in enumerate(zip(r.t, r.raw_x, r.raw_z))]
    del frames[1000:1040]                                    # 40 frames = 200 ms lost mid-recording
    d = dv.StreamDriver(io.BytesIO(b"".join(frames)), INFO); out = d.read()
    assert d.stats.dropped_frames == 40
    S = np.array(acq.SimConfig().S)
    cal = ba.Calibration(S.tolist(), list(acq.SimConfig().offset), 1.0)
    res = pipeline.analyze(out, cal, 750, 200, 300, 1200)
    assert "data_gap" in res["flags"] and not res["usable"]


def test_sequence_and_timestamp_wrap():
    r = rec(5)
    b = dv.simulate_device_bytes(r, INFO, start_seq=65530, t0_us=2 ** 32 - 500_000)
    d = dv.StreamDriver(io.BytesIO(b), INFO); out = d.read()
    assert d.stats.dropped_frames == 0 and len(out.t) == len(r.t)
    assert np.all(np.diff(out.t) > 0) and out.t[-1] == pytest.approx(r.t[-1] - r.t[0], abs=1e-5)


def test_driver_contract_is_abstract():
    with pytest.raises(TypeError):
        dv.SensorDriver()
