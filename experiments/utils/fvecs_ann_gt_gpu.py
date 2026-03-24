#!/usr/bin/env python3
"""
Compute strict (exact) ground truth from fvecs on GPU and save ivecs.

Supports:
1) Full in-memory GPU Flat index (fastest, needs enough VRAM)
2) Streaming exact search fallback (no full-dataset VRAM requirement)

fvecs format:
    For each vector: dim (int32), then dim * float32 values

ivecs format:
    For each vector: k (int32), then k * int32 neighbor IDs
"""

import argparse
import os

import numpy as np


class FvecsMMap:
    """Memory-mapped fvecs reader."""

    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        if self.file_size < 4:
            raise ValueError(f"Invalid fvecs file (too small): {path}")

        with open(path, "rb") as f:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if dim.size != 1:
                raise ValueError(f"Cannot read dim from: {path}")
            self.d = int(dim[0])

        if self.d <= 0:
            raise ValueError(f"Invalid dim={self.d} in {path}")

        rec_bytes = (self.d + 1) * 4
        if self.file_size % rec_bytes != 0:
            raise ValueError(
                f"Invalid fvecs file size for {path}: {self.file_size} not divisible by "
                f"(d+1)*4={(self.d + 1) * 4}"
            )

        self.n = self.file_size // rec_bytes
        self.mm = np.memmap(path, dtype=np.int32, mode="r", shape=(self.n, self.d + 1))

        # Fast sanity checks on first/last rows.
        if self.n > 0 and (self.mm[0, 0] != self.d or self.mm[-1, 0] != self.d):
            raise ValueError(f"Inconsistent vector dim prefix in {path}")

    def get_rows(self, start: int, end: int) -> np.ndarray:
        """Read rows [start, end) as contiguous float32 array of shape (batch, d)."""
        if not (0 <= start <= end <= self.n):
            raise IndexError(f"Invalid range [{start}, {end}) for n={self.n}")
        block_i32 = self.mm[start:end, 1:]
        return block_i32.view(np.float32).copy()


def write_ivecs(path: str, ids: np.ndarray) -> None:
    """Write int32 IDs (nq, k) to ivecs."""
    if ids.dtype != np.int32:
        ids = ids.astype(np.int32, copy=False)
    nq, k = ids.shape
    dims = np.full((nq, 1), k, dtype=np.int32)
    with open(path, "wb") as f:
        np.hstack([dims, ids]).tofile(f)


def metric_from_name(name: str):
    import faiss

    name = name.lower()
    if name == "l2":
        return faiss.METRIC_L2
    if name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {name}")


def build_gpu_flat_index(
    base_reader: FvecsMMap,
    metric: str,
    add_batch_size: int,
    gpu_id: int,
    use_float16: bool,
):
    import faiss

    d = base_reader.d
    metric_type = metric_from_name(metric)

    if metric_type == faiss.METRIC_L2:
        cpu_index = faiss.IndexFlatL2(d)
    else:
        cpu_index = faiss.IndexFlatIP(d)

    res = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = use_float16
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, options)

    print(f"Building exact Flat index on GPU{gpu_id}: dim={d}, metric={metric}")
    print(f"Adding base vectors: n={base_reader.n}, batch={add_batch_size}")
    added = 0
    while added < base_reader.n:
        end = min(added + add_batch_size, base_reader.n)
        xb = base_reader.get_rows(added, end)
        gpu_index.add(xb)
        added = end
        if added % 100000 == 0 or added == base_reader.n:
            print(f"  added {added}/{base_reader.n}", end="\r")
    print()

    return gpu_index


def _merge_topk(prev_D, prev_I, new_D, new_I, k, metric):
    """Merge two top-k result blocks into one top-k block."""
    cand_D = np.concatenate([prev_D, new_D], axis=1)
    cand_I = np.concatenate([prev_I, new_I], axis=1)

    if metric == "l2":
        idx = np.argpartition(cand_D, kth=k - 1, axis=1)[:, :k]
        top_D = np.take_along_axis(cand_D, idx, axis=1)
        top_I = np.take_along_axis(cand_I, idx, axis=1)
        order = np.argsort(top_D, axis=1)
        top_D = np.take_along_axis(top_D, order, axis=1)
        top_I = np.take_along_axis(top_I, order, axis=1)
    else:
        idx = np.argpartition(cand_D, kth=cand_D.shape[1] - k, axis=1)[:, -k:]
        top_D = np.take_along_axis(cand_D, idx, axis=1)
        top_I = np.take_along_axis(cand_I, idx, axis=1)
        order = np.argsort(-top_D, axis=1)
        top_D = np.take_along_axis(top_D, order, axis=1)
        top_I = np.take_along_axis(top_I, order, axis=1)

    return top_D, top_I


def search_streaming_exact_gpu(
    base_reader: FvecsMMap,
    query_reader: FvecsMMap,
    k: int,
    metric: str,
    gpu_id: int,
    use_float16: bool,
    base_chunk_size: int,
    query_batch_size: int,
    out_ivecs: str,
):
    """Exact search by scanning base in chunks on GPU Flat index and merging top-k globally."""
    import faiss

    d = base_reader.d
    nq = query_reader.n
    nb = base_reader.n

    if metric == "l2":
        init_val = np.float32(np.inf)
    else:
        init_val = np.float32(-np.inf)

    top_D = np.full((nq, k), init_val, dtype=np.float32)
    top_I = np.full((nq, k), -1, dtype=np.int32)

    print(
        f"Streaming exact search: nb={nb}, nq={nq}, d={d}, "
        f"base_chunk={base_chunk_size}, query_batch={query_batch_size}"
    )

    chunk_start = 0
    chunk_idx = 0
    total_chunks = (nb + base_chunk_size - 1) // base_chunk_size
    while chunk_start < nb:
        chunk_end = min(chunk_start + base_chunk_size, nb)
        xb = base_reader.get_rows(chunk_start, chunk_end)

        if metric == "l2":
            cpu_index = faiss.IndexFlatL2(d)
        else:
            cpu_index = faiss.IndexFlatIP(d)
        res = faiss.StandardGpuResources()
        opts = faiss.GpuClonerOptions()
        opts.useFloat16 = use_float16
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, opts)
        gpu_index.add(xb)

        q_start = 0
        while q_start < nq:
            q_end = min(q_start + query_batch_size, nq)
            xq = query_reader.get_rows(q_start, q_end)
            D_part, I_part = gpu_index.search(xq, k)
            I_part = I_part.astype(np.int32, copy=False) + chunk_start

            merged_D, merged_I = _merge_topk(
                top_D[q_start:q_end],
                top_I[q_start:q_end],
                D_part.astype(np.float32, copy=False),
                I_part,
                k,
                metric,
            )
            top_D[q_start:q_end] = merged_D
            top_I[q_start:q_end] = merged_I
            q_start = q_end

        chunk_idx += 1
        print(f"  scanned base chunk {chunk_idx}/{total_chunks}", end="\r")
        chunk_start = chunk_end
    print()

    write_ivecs(out_ivecs, top_I)
    print(f"Saved ivecs: {out_ivecs} ({os.path.getsize(out_ivecs)} bytes)")


def search_to_ivecs(
    index,
    query_reader: FvecsMMap,
    k: int,
    search_batch_size: int,
    out_ivecs: str,
):
    nq = query_reader.n
    out_ids = np.empty((nq, k), dtype=np.int32)

    print(f"Searching queries: n={nq}, k={k}, batch={search_batch_size}")
    done = 0
    while done < nq:
        end = min(done + search_batch_size, nq)
        xq = query_reader.get_rows(done, end)
        _, I = index.search(xq, k)
        out_ids[done:end] = I.astype(np.int32, copy=False)
        done = end
        if done % 100000 == 0 or done == nq:
            print(f"  searched {done}/{nq}", end="\r")
    print()

    write_ivecs(out_ivecs, out_ids)
    print(f"Saved ivecs: {out_ivecs} ({os.path.getsize(out_ivecs)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute exact ground truth from fvecs on GPU and save ivecs."
    )
    parser.add_argument("--base_fvecs", required=True, help="Base vectors in fvecs format")
    parser.add_argument("--query_fvecs", required=True, help="Query vectors in fvecs format")
    parser.add_argument("--output_ivecs", required=True, help="Output ivecs path")
    parser.add_argument("--k", type=int, default=100, help="Top-k neighbors (default: 100)")
    parser.add_argument(
        "--metric",
        choices=["l2", "ip"],
        default="l2",
        help="Distance metric (default: l2)",
    )
    parser.add_argument(
        "--add_batch_size",
        type=int,
        default=200000,
        help="Batch size when adding base vectors to flat index (default: 200000)",
    )
    parser.add_argument(
        "--search_batch_size",
        type=int,
        default=10000,
        help="Batch size when searching queries (default: 10000)",
    )
    parser.add_argument(
        "--base_chunk_size",
        type=int,
        default=100000,
        help="Base chunk size for streaming exact fallback (default: 100000)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "stream"],
        default="auto",
        help="full=single full GPU Flat index; stream=streaming exact; auto=try full then fallback",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument(
        "--use_float16",
        action="store_true",
        help="Use float16 on GPU to reduce memory",
    )
    args = parser.parse_args()

    base = FvecsMMap(args.base_fvecs)
    query = FvecsMMap(args.query_fvecs)
    if base.d != query.d:
        raise ValueError(f"Dimension mismatch: base d={base.d}, query d={query.d}")
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.k > base.n:
        raise ValueError(f"--k ({args.k}) cannot exceed base size ({base.n})")

    print(f"Base : {args.base_fvecs} -> n={base.n}, d={base.d}")
    print(f"Query: {args.query_fvecs} -> n={query.n}, d={query.d}")
    print(
        f"Config: metric={args.metric}, k={args.k}, mode={args.mode}, "
        f"gpu_id={args.gpu_id}, fp16={args.use_float16}"
    )

    full_attempted = args.mode in ("auto", "full")
    if full_attempted:
        try:
            index = build_gpu_flat_index(
                base_reader=base,
                metric=args.metric,
                add_batch_size=args.add_batch_size,
                gpu_id=args.gpu_id,
                use_float16=args.use_float16,
            )
            search_to_ivecs(
                index=index,
                query_reader=query,
                k=args.k,
                search_batch_size=args.search_batch_size,
                out_ivecs=args.output_ivecs,
            )
            return
        except RuntimeError as e:
            if args.mode == "full":
                raise
            err = str(e).lower()
            if "out of memory" in err or "alloc fail" in err or "cudamalloc" in err:
                print("Full GPU Flat index OOM, fallback to streaming exact mode.")
            else:
                raise

    search_streaming_exact_gpu(
        base_reader=base,
        query_reader=query,
        k=args.k,
        metric=args.metric,
        gpu_id=args.gpu_id,
        use_float16=args.use_float16,
        base_chunk_size=args.base_chunk_size,
        query_batch_size=args.search_batch_size,
        out_ivecs=args.output_ivecs,
    )


if __name__ == "__main__":
    main()
