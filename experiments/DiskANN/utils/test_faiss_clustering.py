#!/usr/bin/env python3
import argparse
import os
import struct
import sys

import numpy as np

try:
    import faiss
except ImportError as exc:
    print(f"Failed to import faiss: {exc}", file=sys.stderr)
    sys.exit(1)


def read_fvecs_head(path: str, count: int) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise RuntimeError(f"{path}: failed to read dimension header")
        dim = struct.unpack("i", header)[0]
        if dim <= 0:
            raise RuntimeError(f"{path}: invalid dimension {dim}")

        file_size = os.fstat(f.fileno()).st_size
        row_bytes = (dim + 1) * 4
        if file_size % row_bytes != 0:
            raise RuntimeError(f"{path}: invalid fvecs file size")
        total = file_size // row_bytes
        n = min(count, total)

        f.seek(0)
        raw = np.fromfile(f, dtype=np.int32, count=n * (dim + 1))
        if raw.size != n * (dim + 1):
            raise RuntimeError(f"{path}: failed to read {n} vectors")

    raw = raw.reshape(n, dim + 1)
    dims = raw[:, 0]
    if np.any(dims != dim):
        bad = int(np.nonzero(dims != dim)[0][0])
        raise RuntimeError(f"{path}: inconsistent dimension at vector {bad}")
    return raw[:, 1:].view(np.float32).copy()


def read_fbin_head(path: str, count: int) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{path}: failed to read fbin header")
        n_total, dim = struct.unpack("ii", header)
        if n_total <= 0 or dim <= 0:
            raise RuntimeError(f"{path}: invalid header n={n_total}, d={dim}")

        n = min(count, n_total)
        data = np.fromfile(f, dtype=np.float32, count=n * dim)
        if data.size != n * dim:
            raise RuntimeError(f"{path}: failed to read {n} vectors")
    return data.reshape(n, dim)


def read_u8bin_head(path: str, count: int) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{path}: failed to read u8bin header")
        n_total, dim = struct.unpack("ii", header)
        if n_total <= 0 or dim <= 0:
            raise RuntimeError(f"{path}: invalid header n={n_total}, d={dim}")

        n = min(count, n_total)
        data = np.fromfile(f, dtype=np.uint8, count=n * dim)
        if data.size != n * dim:
            raise RuntimeError(f"{path}: failed to read {n} vectors")
    return data.reshape(n, dim).astype(np.float32, copy=False)


def infer_format(path: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    lower = path.lower()
    if lower.endswith(".fvecs"):
        return "fvecs"
    if lower.endswith(".u8bin"):
        return "u8bin"
    if lower.endswith(".fbin") or lower.endswith(".bin"):
        return "fbin"
    raise RuntimeError("Cannot infer format; pass --format explicitly")


def load_vectors(path: str, fmt: str, count: int) -> np.ndarray:
    fmt = infer_format(path, fmt)
    if fmt == "fvecs":
        return read_fvecs_head(path, count)
    if fmt == "fbin":
        return read_fbin_head(path, count)
    if fmt == "u8bin":
        return read_u8bin_head(path, count)
    raise RuntimeError(f"Unsupported format: {fmt}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read the first N vectors and run FAISS Clustering.train()"
    )
    parser.add_argument("input", help="Input vector file")
    parser.add_argument("--format", default="auto", choices=["auto", "fvecs", "fbin", "u8bin"])
    parser.add_argument("--sample-num", type=int, default=200000, help="Number of vectors to read")
    parser.add_argument("--clusters", type=int, default=512, help="Number of centroids")
    parser.add_argument("--niter", type=int, default=25, help="KMeans iterations")
    parser.add_argument("--seed", type=int, default=1234, help="FAISS clustering seed")
    parser.add_argument("--verbose", action="store_true", help="Enable FAISS clustering logs")
    args = parser.parse_args()

    xb = load_vectors(args.input, args.format, args.sample_num)
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    n, d = xb.shape

    if args.clusters <= 0:
        raise RuntimeError("--clusters must be > 0")
    if n < args.clusters:
        raise RuntimeError(f"Need at least clusters vectors: n={n}, clusters={args.clusters}")

    print(f"Loaded vectors: n={n}, d={d}, dtype={xb.dtype}")
    print(
        f"Running faiss.Clustering.train with clusters={args.clusters}, "
        f"niter={args.niter}, seed={args.seed}"
    )

    cp = faiss.ClusteringParameters()
    cp.niter = args.niter
    cp.seed = args.seed
    cp.verbose = args.verbose

    clus = faiss.Clustering(d, args.clusters, cp)
    index = faiss.IndexFlatL2(d)
    clus.train(xb, index)

    centroids = faiss.vector_to_array(clus.centroids).reshape(args.clusters, d)
    print("Clustering finished successfully")
    print(f"Centroids shape: {centroids.shape}")
    print(f"First centroid L2 norm: {np.linalg.norm(centroids[0]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
