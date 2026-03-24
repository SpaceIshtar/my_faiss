#!/usr/bin/env python3
"""Convert .fvecs (per-vector dim header) to DiskANN .bin format.

Output .bin layout:
  int32 npts
  int32 dim
  float32 data[npts * dim]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert fvecs to DiskANN bin")
    p.add_argument("input", type=Path, help="Input .fvecs path")
    p.add_argument("output", type=Path, help="Output .bin path")
    p.add_argument(
        "--block-vectors",
        type=int,
        default=65536,
        help="Vectors processed per block (default: 65536)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = args.input
    out_path = args.output

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        return 1

    if out_path.exists() and not args.overwrite:
        print(f"[ERROR] Output exists: {out_path} (use --overwrite)", file=sys.stderr)
        return 1

    file_size = in_path.stat().st_size
    if file_size < 4:
        print("[ERROR] Input file too small", file=sys.stderr)
        return 1

    with in_path.open("rb") as fin:
        first4 = fin.read(4)
        if len(first4) != 4:
            print("[ERROR] Failed to read first dimension", file=sys.stderr)
            return 1
        (dim,) = struct.unpack("<i", first4)
        if dim <= 0:
            print(f"[ERROR] Invalid dim in fvecs header: {dim}", file=sys.stderr)
            return 1

    record_bytes = (dim + 1) * 4
    if file_size % record_bytes != 0:
        print(
            f"[ERROR] File size mismatch for dim={dim}: size={file_size}, "
            f"record_bytes={record_bytes}",
            file=sys.stderr,
        )
        return 1

    npts = file_size // record_bytes
    if npts > 2_147_483_647:
        print(f"[ERROR] npts exceeds int32: {npts}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    block_vectors = max(1, int(args.block_vectors))
    in_block_bytes = block_vectors * record_bytes

    print(f"[INFO] input={in_path}")
    print(f"[INFO] output={out_path}")
    print(f"[INFO] npts={npts}, dim={dim}, file_size={file_size}")

    with in_path.open("rb") as fin, out_path.open("wb") as fout:
        fout.write(struct.pack("<ii", int(npts), int(dim)))

        converted = 0
        while converted < npts:
            raw = fin.read(in_block_bytes)
            if not raw:
                break

            if len(raw) % record_bytes != 0:
                print("[ERROR] Read partial record; input seems corrupted", file=sys.stderr)
                return 1

            cur_n = len(raw) // record_bytes
            out_block = bytearray(cur_n * dim * 4)

            for i in range(cur_n):
                in_off = i * record_bytes
                (cur_dim,) = struct.unpack_from("<i", raw, in_off)
                if cur_dim != dim:
                    print(
                        f"[ERROR] Inconsistent dim at vector {converted + i}: "
                        f"expected {dim}, got {cur_dim}",
                        file=sys.stderr,
                    )
                    return 1

                src = in_off + 4
                dst = i * dim * 4
                out_block[dst : dst + dim * 4] = raw[src : src + dim * 4]

            fout.write(out_block)
            converted += cur_n

            if converted % (block_vectors * 10) == 0 or converted == npts:
                pct = 100.0 * converted / npts
                print(f"[INFO] converted {converted}/{npts} ({pct:.2f}%)")

    out_size = out_path.stat().st_size
    expected_out = 8 + npts * dim * 4
    if out_size != expected_out:
        print(
            f"[ERROR] Output size mismatch: got {out_size}, expected {expected_out}",
            file=sys.stderr,
        )
        return 1

    print("[OK] Conversion completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
