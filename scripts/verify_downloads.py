"""Verify GaitRec source files against the sizes and MD5 checksums published by figshare.

Usage:  python scripts/verify_downloads.py [data_dir]
Exit code 0 only if every required file is present and intact.
Do not delete the raw files until this passes AND feature extraction has been validated
AND the waveform baseline (M1) has been fitted.
"""
import hashlib
import sys
from pathlib import Path

# name: (bytes, md5, figshare download URL) -- from api.figshare.com, collection 4788012
SOURCES = {
    "GRF_metadata.csv":       (629_530,     "49305e34d3402f1dd988bb9cd1e0fdc0", "https://ndownloader.figshare.com/files/22062960"),
    "GRF_F_V_RAW_left.csv":   (254_589_160, "e659dc91ad9c6d15063c20a384f8ddcf", "https://ndownloader.figshare.com/files/22063197"),
    "GRF_F_V_RAW_right.csv":  (253_659_212, "f596cd1a3df138beda9fd99125dbb0dd", "https://ndownloader.figshare.com/files/22063200"),
    "GRF_F_AP_RAW_left.csv":  (261_842_540, "0f373f27ad093bacfbb8db30e137bbe7", "https://ndownloader.figshare.com/files/22063023"),
    "GRF_F_AP_RAW_right.csv": (260_995_350, "70e82d0b50493201f277780c6d2ad4ec", "https://ndownloader.figshare.com/files/22063110"),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(data_dir: str = "data") -> int:
    ok = True
    for name, (size, digest, url) in SOURCES.items():
        p = Path(data_dir) / name
        if not p.exists():
            print(f"MISSING  {name}  -> download from {url}"); ok = False; continue
        if p.stat().st_size != size:
            print(f"SIZE     {name}  {p.stat().st_size} bytes, expected {size} (incomplete download?)"); ok = False; continue
        if md5(p) != digest:
            print(f"CHECKSUM {name}  does not match figshare"); ok = False; continue
        print(f"OK       {name}")
    print("\nAll files verified." if ok else "\nVerification FAILED. Do not run extraction or delete anything.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
