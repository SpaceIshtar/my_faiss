#!/usr/bin/env python3
"""
Convert youtube-8m audio/video embeddings from raw float64 to fvecs,
and compute ground truth (L2) using GPU FAISS.

Input format:  n * dim * sizeof(float64), no header
Output format: fvecs (int32 dim + dim * float32 per vector)
               ivecs (int32 k + k * int32 per query)
"""

import numpy as np
import os
import faiss


def f64_to_fvecs(input_path, output_path, dim, chunk_size=100000):
    """Convert raw float64 file to fvecs (float32)."""
    file_size = os.path.getsize(input_path)
    n = file_size // (dim * 8)
    assert file_size == n * dim * 8, f"File size {file_size} not divisible by {dim}*8"

    print(f"Converting {input_path} -> {output_path}")
    print(f"  {n} vectors, {dim} dims, float64 -> float32")

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        written = 0
        while written < n:
            batch = min(chunk_size, n - written)
            data = np.fromfile(fin, dtype=np.float64, count=batch * dim)
            data = data.reshape(batch, dim).astype(np.float32)
            dims = np.full((batch, 1), dim, dtype=np.int32)
            np.hstack([dims.view(np.float32), data]).tofile(fout)
            written += batch
            if written % 500000 == 0 or written == n:
                print(f"  {written}/{n}", end="\r")
        print()
    print(f"  Done. {os.path.getsize(output_path)} bytes")


def compute_gt_gpu(base_path, query_path, dim, ivecs_path, fvecs_path, k=100):
    """Load raw float64 base/query, compute L2 ground truth on GPU."""
    base_size = os.path.getsize(base_path)
    query_size = os.path.getsize(query_path)
    n_base = base_size // (dim * 8)
    n_query = query_size // (dim * 8)

    print(f"Computing L2 ground truth (GPU)")
    print(f"  Base: {n_base} x {dim}, Query: {n_query} x {dim}, k={k}")

    print("  Loading base...")
    with open(base_path, "rb") as f:
        base = np.fromfile(f, dtype=np.float64, count=n_base * dim)
        base = base.reshape(n_base, dim).astype(np.float32)

    print("  Loading query...")
    with open(query_path, "rb") as f:
        queries = np.fromfile(f, dtype=np.float64, count=n_query * dim)
        queries = queries.reshape(n_query, dim).astype(np.float32)

    print("  Building GPU index...")
    cpu_index = faiss.IndexFlatL2(dim)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    index.add(base)

    print("  Searching...")
    D, I = index.search(queries, k)

    ids = I.astype(np.int32)
    print(f"  ID range: [{ids.min()}, {ids.max()}]")
    print(f"  Dist range: [{D.min():.4f}, {D.max():.4f}]")

    dims_col = np.full((n_query, 1), k, dtype=np.int32)

    with open(ivecs_path, "wb") as f:
        np.hstack([dims_col, ids]).tofile(f)
    print(f"  IDs -> {ivecs_path}")

    with open(fvecs_path, "wb") as f:
        np.hstack([dims_col.view(np.float32), D]).tofile(f)
    print(f"  Dists -> {fvecs_path}")


def process_dataset(name, dim, src_dir, out_dir):
    """Process one dataset (audio or video)."""
    print(f"\n{'='*60}")
    print(f"Processing {name} (dim={dim})")
    print(f"{'='*60}")

    base_raw = os.path.join(src_dir, f"yt8m_{name}_embedding.fvecs")
    query_raw = os.path.join(src_dir, f"yt8m_{name}_querys_10k.fvecs")

    os.makedirs(out_dir, exist_ok=True)

    base_out = os.path.join(out_dir, f"{name}_base.fvecs")
    query_out = os.path.join(out_dir, f"{name}_query_10k.fvecs")
    gt_ivecs = os.path.join(out_dir, f"{name}_gt100.ivecs")
    gt_fvecs = os.path.join(out_dir, f"{name}_gt100_dist.fvecs")

    f64_to_fvecs(base_raw, base_out, dim)
    f64_to_fvecs(query_raw, query_out, dim)
    compute_gt_gpu(base_raw, query_raw, dim, gt_ivecs, gt_fvecs, k=100)


if __name__ == "__main__":
    src_dir = "/data/local/embedding_dataset/youtube-8m/embedding_metadata_anns"

    process_dataset("audio", 128,
                    src_dir, os.path.join(src_dir, "audio"))
    process_dataset("video", 1024,
                    src_dir, os.path.join(src_dir, "video"))

    print("\nAll done.")
