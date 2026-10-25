# GaitSense sensor-driver interface (v1, draft)

Purpose: let real hardware replace the simulator without touching analysis code. Not yet exercised on a physical device.

## Software contract (`gaitsense.drivers.SensorDriver`)
- `info() -> DeviceInfo`: serial, firmware, nominal `fs_hz`, `lsb_x`/`lsb_z` (raw units per ADC count), limb side.
- `read() -> Recording`: `t` in seconds from one clock shared by both channels, strictly increasing; `raw_x`, `raw_z` in
  the same units the calibration was fitted in; optional `start_time_h` and `temp_c` if the drift model is used.
- A driver must not repair, interpolate or fill data. Lost samples must appear as gaps so the quality gate can see them.

## Wire protocol v1 (reference implementation: `FrameParser`)
18-byte frames: magic `A5 5A`, seq u16, t_us u32, x i32, z i32, CRC-16/CCITT-FALSE (init 0xFFFF, poly 0x1021) over the first 16 bytes. Little-endian.
- CRC failure: skip one byte and resynchronise on the next magic; count `crc_errors`.
- Sequence gaps are counted as `dropped_frames`. Sequence and time wrap are handled.
- Time unwrap assumes kept frames are less than 35 minutes apart.

## Firmware requirements
1. Sample both channels at the same instant (or document the skew; the software assumes none).
2. Timestamp from a crystal-referenced clock, not from the host receive time.
3. Report ADC resolution, reference voltage, full scale, and gain so `lsb_x`/`lsb_z` can be derived and checked.
4. Provide a command to apply the built-in reference load (for drift checks) and log temperature with each check.

## To validate on real hardware
Measured clock accuracy and jitter; channel skew; frame loss at the real baud/bluetooth rate; behaviour on cable pull and
low battery. None of this has been measured.
