#!/usr/bin/env python3
"""
Convert malformed old fvecs (raw float64 blocks, no per-vector dim prefix)
to standard fvecs format.

Old format:
    n * dim * sizeof(float64), no header

New format (standard fvecs):
    For each vector: dim (int32), then dim * float32 values
"""

import argparse
import os

import numpy as np


def old_f64_block_to_fvecs(input_path, output_path, dim=2048, chunk_size=10000):
    """Convert raw float64 block file to standard fvecs."""
    file_size = os.path.getsize(input_path)
    vec_bytes = dim * 8  # float64
    if file_size % vec_bytes != 0:
        raise ValueError(
            f"Invalid old file size: {file_size} bytes is not divisible by {vec_bytes} "
            f"(dim={dim}, float64)."
        )

    n = file_size // vec_bytes
    expected_out_size = n * (4 + dim * 4)  # int32 dim + dim float32

    print(f"Converting: {input_path}")
    print(f"       into: {output_path}")
    print(f"  vectors={n}, dim={dim}, dtype: float64 -> float32")
    print(f"  expected output size: {expected_out_size} bytes")

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        written = 0
        while written < n:
            batch = min(chunk_size, n - written)
            data = np.fromfile(fin, dtype=np.float64, count=batch * dim)
            if data.size != batch * dim:
                raise RuntimeError(
                    f"Unexpected EOF while reading {input_path}. "
                    f"Expected {batch * dim} float64 values, got {data.size}."
                )

            data = data.reshape(batch, dim).astype(np.float32, copy=False)
            dims = np.full((batch, 1), dim, dtype=np.int32)
            out = np.hstack([dims.view(np.float32), data])
            out.tofile(fout)

            written += batch
            if written % 100000 == 0 or written == n:
                print(f"  progress: {written}/{n}", end="\r")
        print()

    real_out_size = os.path.getsize(output_path)
    if real_out_size != expected_out_size:
        raise RuntimeError(
            f"Output size mismatch: {real_out_size} != {expected_out_size}."
        )
    print(f"Done. Output size: {real_out_size} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Convert malformed float64-block fvecs to standard fvecs."
    )
    parser.add_argument("--input", required=True, help="Path to old malformed fvecs file")
    parser.add_argument("--output", required=True, help="Path to new standard fvecs file")
    parser.add_argument(
        "--dim",
        type=int,
        default=2048,
        help="Vector dimension in old file (default: 2048)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Vectors per batch while converting (default: 10000)",
    )
    args = parser.parse_args()

    old_f64_block_to_fvecs(
        input_path=args.input,
        output_path=args.output,
        dim=args.dim,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
