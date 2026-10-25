"""Hardware integration (step 13 interface). See docs/driver_interface.md for the contract.

Wire protocol v1 (little-endian), one frame per sample pair, 18 bytes:
    magic  2B  0xA5 0x5A
    seq    u16 increments by 1 per frame, wraps at 65536
    t_us   u32 device microsecond clock, wraps at 2**32 (~71.6 min)
    x      i32 fore-aft ADC counts (signed)
    z      i32 vertical ADC counts (signed)
    crc    u16 CRC-16/CCITT-FALSE over the preceding 16 bytes
Bad frames are dropped, never repaired. Dropped frames show up as time gaps, which the quality gate flags.
"""
from __future__ import annotations
import binascii
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
import numpy as np
from .acquisition import Recording

MAGIC = b"\xa5\x5a"
FRAME_FMT = "<2sHIii"
FRAME_LEN = struct.calcsize(FRAME_FMT) + 2   # 18


def crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


def encode_frame(seq: int, t_us: int, x: int, z: int) -> bytes:
    body = struct.pack(FRAME_FMT, MAGIC, seq & 0xFFFF, t_us & 0xFFFFFFFF, x, z)
    return body + struct.pack("<H", crc16(body))


@dataclass
class DeviceInfo:
    serial: str
    firmware: str
    fs_hz: float
    lsb_x: float        # raw units per ADC count, fore-aft (e.g. volts)
    lsb_z: float
    side: str = "unknown"


@dataclass
class DriverStats:
    frames_ok: int = 0
    crc_errors: int = 0
    bytes_discarded: int = 0
    dropped_frames: int = 0        # inferred from sequence gaps
    truncated_tail_bytes: int = 0


class FrameParser:
    """Stream parser with resynchronisation. Feed bytes in any chunking."""
    def __init__(self):
        self.buf = bytearray(); self.stats = DriverStats()
        self._last_seq: Optional[int] = None

    def feed(self, data: bytes) -> Iterator[Tuple[int, int, int, int]]:
        self.buf += data
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                keep = 1 if self.buf.endswith(MAGIC[:1]) else 0
                self.stats.bytes_discarded += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]; return
            if i > 0:
                self.stats.bytes_discarded += i; del self.buf[:i]
            if len(self.buf) < FRAME_LEN:
                return
            frame = bytes(self.buf[:FRAME_LEN])
            if crc16(frame[:-2]) != struct.unpack("<H", frame[-2:])[0]:
                self.stats.crc_errors += 1; self.stats.bytes_discarded += 1
                del self.buf[:1]; continue          # resync: skip one byte and search for the next magic
            _, seq, t_us, x, z = struct.unpack(FRAME_FMT, frame[:-2])
            if self._last_seq is not None:
                self.stats.dropped_frames += (seq - self._last_seq - 1) & 0xFFFF
            self._last_seq = seq
            self.stats.frames_ok += 1
            del self.buf[:FRAME_LEN]
            yield seq, t_us, x, z

    def finish(self):
        self.stats.truncated_tail_bytes = len(self.buf)


class SensorDriver(ABC):
    """Contract every hardware driver must satisfy. `read()` returns a Recording on the SAME shared clock for both
    channels, in seconds, strictly increasing; raw values in the units the calibration was fitted in."""
    @abstractmethod
    def info(self) -> DeviceInfo: ...
    @abstractmethod
    def read(self) -> Recording: ...


class StreamDriver(SensorDriver):
    """Reads the v1 protocol from any object with .read(n) -> bytes (pyserial Serial, socket file, BytesIO).
    Reads until the stream returns no bytes. Not real-time: it collects a finite recording."""
    def __init__(self, stream, device_info: DeviceInfo, chunk: int = 4096):
        self.stream, self._info, self.chunk = stream, device_info, chunk
        self.stats = DriverStats()

    def info(self) -> DeviceInfo:
        return self._info

    def read(self) -> Recording:
        p = FrameParser(); ts: List[int] = []; xs: List[int] = []; zs: List[int] = []
        while True:
            data = self.stream.read(self.chunk)
            if not data:
                break
            for _, t_us, x, z in p.feed(data):
                ts.append(t_us); xs.append(x); zs.append(z)
        p.finish(); self.stats = p.stats
        if not ts:
            return Recording([], [], [], side=self._info.side, source=f"device:{self._info.serial}")
        t = _unwrap_u32(np.array(ts, dtype=np.uint64)) * 1e-6
        i = self._info
        return Recording(t - t[0], np.array(xs) * i.lsb_x, np.array(zs) * i.lsb_z, side=i.side,
                         source=f"device:{i.serial} fw{i.firmware}")


def _unwrap_u32(t: np.ndarray) -> np.ndarray:
    """Unwrap a u32 microsecond clock. Assumes consecutive kept frames are < 2**31 us (35 min) apart."""
    t = t.astype(np.int64)
    d = np.diff(t)
    d = np.where(d < -(2 ** 31), d + 2 ** 32, d)
    return np.concatenate([[t[0]], t[0] + np.cumsum(d)]).astype(float)


def simulate_device_bytes(rec: Recording, info: DeviceInfo, start_seq: int = 0, t0_us: int = 0) -> bytes:
    """Encode a Recording as device bytes (for tests/dry runs). Inverse of StreamDriver for clean streams."""
    out = bytearray()
    for k, (t, x, z) in enumerate(zip(rec.t, rec.raw_x, rec.raw_z)):
        out += encode_frame(start_seq + k, t0_us + int(round(t * 1e6)), int(round(x / info.lsb_x)), int(round(z / info.lsb_z)))
    return bytes(out)
