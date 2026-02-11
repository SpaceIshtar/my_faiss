#!/usr/bin/env python3
"""
Convert BigANN benchmark binary files (.fbin) to fvecs/ivecs format.

BigANN .fbin format:
    Header: num_points (uint32), num_dims (uint32)
    Data:   num_points * num_dims * sizeof(float32)

BigANN ground truth format (.fbin with k neighbors per query):
    Header: num_queries (uint32), k (uint32)
    Data:   Block layout — all n*k int32 IDs first, then all n*k float32 distances

fvecs format: For each vector: dim (int32), then dim float32 values
ivecs format: For each vector: dim (int32), then dim int32 values
"""

import argparse
import numpy as np
import os
import struct
import sys


def read_fbin_header(path):
    """Read the header of a BigANN .fbin file."""
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    return n, d


def fbin_to_fvecs(input_path, output_path, chunk_size=10000):
    """Convert a BigANN .fbin float file to fvecs format."""
    n, d = read_fbin_header(input_path)
    file_size = os.path.getsize(input_path)
    expected = 8 + n * d * 4
    assert file_size == expected, (
        f"File size mismatch: {file_size} != {expected} "
        f"(n={n}, d={d})"
    )

    print(f"Converting {input_path} -> {output_path}")
    print(f"  {n} vectors, {d} dimensions, float32")

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        fin.seek(8)  # skip header
        written = 0
        while written < n:
            batch = min(chunk_size, n - written)
            data = np.fromfile(fin, dtype=np.float32, count=batch * d)
            data = data.reshape(batch, d)
            # Prepend dimension to each vector
            dims = np.full((batch, 1), d, dtype=np.int32)
            out = np.hstack([dims.view(np.float32), data])
            out.tofile(fout)
            written += batch
            if written % 100000 == 0 or written == n:
                print(f"  {written}/{n} vectors written", end="\r")
        print()

    print(f"  Done. Output: {output_path} ({os.path.getsize(output_path)} bytes)")


def gt_fbin_to_ivecs_fvecs(input_path, ivecs_path, fvecs_path=None, chunk_size=10000):
    """Convert BigANN ground truth file to ivecs (IDs) and optionally fvecs (distances).

    Ground truth block layout: all n*k int32 IDs first, then all n*k float32 distances.
    """
    n, k = read_fbin_header(input_path)
    file_size = os.path.getsize(input_path)
    expected = 8 + n * k * 2 * 4  # n*k IDs (int32) + n*k distances (float32)
    assert file_size == expected, (
        f"File size mismatch: {file_size} != {expected} "
        f"(n={n}, k={k}). Is this a ground truth file?"
    )

    print(f"Converting ground truth {input_path}")
    print(f"  {n} queries, {k} neighbors each")

    with open(input_path, "rb") as fin:
        fin.seek(8)
        # Block layout: read all IDs, then all distances
        all_ids = np.fromfile(fin, dtype=np.int32, count=n * k).reshape(n, k)
        all_dists = np.fromfile(fin, dtype=np.float32, count=n * k).reshape(n, k)

    print(f"  ID range: [{all_ids.min()}, {all_ids.max()}]")
    print(f"  Distance range: [{all_dists.min():.4f}, {all_dists.max():.4f}]")

    # Write ivecs: dim (int32) + k int32 IDs per query
    with open(ivecs_path, "wb") as fout:
        written = 0
        while written < n:
            batch = min(chunk_size, n - written)
            ids = all_ids[written:written + batch]
            dims = np.full((batch, 1), k, dtype=np.int32)
            out = np.hstack([dims, ids])
            out.tofile(fout)
            written += batch

    # Write fvecs: dim (int32) + k float32 distances per query
    if fvecs_path:
        with open(fvecs_path, "wb") as fout:
            written = 0
            while written < n:
                batch = min(chunk_size, n - written)
                dists = all_dists[written:written + batch]
                dims = np.full((batch, 1), k, dtype=np.int32)
                out = np.hstack([dims.view(np.float32), dists])
                out.tofile(fout)
                written += batch

    print(f"  IDs -> {ivecs_path} ({os.path.getsize(ivecs_path)} bytes)")
    if fvecs_path:
        print(f"  Distances -> {fvecs_path} ({os.path.getsize(fvecs_path)} bytes)")


def compute_gt_bruteforce(base_path, query_path, ivecs_path, fvecs_path=None, k=100):
    """Recompute ground truth by brute-force inner product search against the base."""
    import faiss

    n_base, d = read_fbin_header(base_path)
    n_query, d_q = read_fbin_header(query_path)
    assert d == d_q, f"Dimension mismatch: base d={d}, query d={d_q}"

    print(f"Computing brute-force ground truth (L2)")
    print(f"  Base: {n_base} x {d}, Query: {n_query} x {d_q}, k={k}")

    with open(base_path, "rb") as f:
        f.seek(8)
        base = np.fromfile(f, dtype=np.float32, count=n_base * d).reshape(n_base, d)
    with open(query_path, "rb") as f:
        f.seek(8)
        queries = np.fromfile(f, dtype=np.float32, count=n_query * d).reshape(n_query, d)

    cpu_index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_all_gpus(cpu_index)
    index.add(base)
    print("  Searching (GPU)...")
    D, I = index.search(queries, k)

    all_ids = I.astype(np.int32)
    print(f"  ID range: [{all_ids.min()}, {all_ids.max()}]")
    print(f"  Distance range: [{D.min():.4f}, {D.max():.4f}]")

    # Write ivecs
    dims = np.full((n_query, 1), k, dtype=np.int32)
    with open(ivecs_path, "wb") as f:
        np.hstack([dims, all_ids]).tofile(f)
    print(f"  IDs -> {ivecs_path}")

    # Write distance fvecs
    if fvecs_path:
        with open(fvecs_path, "wb") as f:
            np.hstack([dims.view(np.float32), D]).tofile(f)
        print(f"  Distances -> {fvecs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BigANN .fbin files to fvecs/ivecs format"
    )
    parser.add_argument(
        "--input_dir",
        default="/data/local/embedding_dataset/text2image",
        help="Directory containing .fbin files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as input_dir)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    os.makedirs(output_dir, exist_ok=True)

    # 1. Base vectors: fbin -> fvecs
    base_in = os.path.join(input_dir, "base.1B.fbin.crop_nb_1000000")
    base_out = os.path.join(output_dir, "base_1M.fvecs")
    if os.path.exists(base_in):
        fbin_to_fvecs(base_in, base_out)
    else:
        print(f"Skipping (not found): {base_in}")

    # 2. Query vectors: fbin -> fvecs
    query_in = os.path.join(input_dir, "query.heldout.30K.fbin")
    query_out = os.path.join(output_dir, "query_30K.fvecs")
    if os.path.exists(query_in):
        fbin_to_fvecs(query_in, query_out)
    else:
        print(f"Skipping (not found): {query_in}")

    # 3. Recompute ground truth for the 1M crop via brute-force
    gt_ivecs_out = os.path.join(output_dir, "gt100_30K.ivecs")
    gt_fvecs_out = os.path.join(output_dir, "gt100_30K_dist.fvecs")
    compute_gt_bruteforce(base_in, query_in, gt_ivecs_out, gt_fvecs_out, k=100)

    print("\nAll conversions complete.")


if __name__ == "__main__":
    main()
