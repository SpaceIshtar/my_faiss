"""
Benchmark FAISS IndexHNSWSQ with exact-distance reranking.

This script builds an IndexHNSWSQ (HNSW graph with SQ storage) and wraps it
with IndexRefine. For each ef_search value, HNSW traverses the graph using
SQ distances to collect ef candidates, then reranks them with exact L2
distances and returns the top-k results.

Usage:
    python test_hnswsq.py --dataset sift1m
    python test_hnswsq.py --dataset gist1M --threads 8
    python test_hnswsq.py --dataset sift1m --ef 10,20,50,100
"""

import argparse
import configparser
import os
import struct
import time

import faiss
import numpy as np


# ── Config parsing ─────────────────────────────────────────────

def parse_datasets_conf(filepath):
    """Parse datasets.conf (INI-style) into a dict of dataset configs."""
    cp = configparser.ConfigParser()
    cp.read(filepath)
    datasets = {}
    for section in cp.sections():
        cfg = dict(cp[section])
        for int_key in ("threads", "k", "hnsw_M", "hnsw_efConstruction"):
            if int_key in cfg:
                cfg[int_key] = int(cfg[int_key])
        if "ef_search" in cfg:
            cfg["ef_search"] = [int(x.strip()) for x in cfg["ef_search"].split(",")]
        datasets[section] = cfg
    return datasets


def parse_sq_conf(filepath):
    """Parse sq.conf into a dict mapping dataset -> list of qtype strings."""
    param_sets = {}
    current_section = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                param_sets.setdefault(current_section, [])
            elif current_section is not None and line.startswith("qtype="):
                qtype_str = line.split("=", 1)[1].strip()
                param_sets[current_section].append(qtype_str)

    return param_sets


# ── Data I/O (fvecs / ivecs) ──────────────────────────────────

def fvecs_read(filepath):
    """Read .fvecs file, return numpy float32 array of shape (n, d)."""
    with open(filepath, "rb") as f:
        buf = f.read()

    d = struct.unpack_from("<i", buf, 0)[0]
    row_bytes = 4 + d * 4
    n = len(buf) // row_bytes

    data = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        offset = i * row_bytes + 4
        data[i] = np.frombuffer(buf, dtype=np.float32, count=d, offset=offset)
    return data


def ivecs_read(filepath):
    """Read .ivecs file, return numpy int32 array of shape (n, d)."""
    with open(filepath, "rb") as f:
        buf = f.read()

    d = struct.unpack_from("<i", buf, 0)[0]
    row_bytes = 4 + d * 4
    n = len(buf) // row_bytes

    data = np.zeros((n, d), dtype=np.int32)
    for i in range(n):
        offset = i * row_bytes + 4
        data[i] = np.frombuffer(buf, dtype=np.int32, count=d, offset=offset)
    return data


# ── Recall computation ────────────────────────────────────────

def compute_recall(result_ids, gt, k):
    """Compute recall@k. result_ids: (nq, k), gt: (nq, gt_k)."""
    nq = result_ids.shape[0]
    hits = 0
    for i in range(nq):
        gt_set = set(gt[i, :k].tolist())
        for j in range(k):
            if result_ids[i, j] in gt_set:
                hits += 1
    return hits / (nq * k)


# ── qtype string -> faiss enum ────────────────────────────────

QTYPE_MAP = {
    "QT_8bit":          faiss.ScalarQuantizer.QT_8bit,
    "8bit":             faiss.ScalarQuantizer.QT_8bit,
    "QT_4bit":          faiss.ScalarQuantizer.QT_4bit,
    "4bit":             faiss.ScalarQuantizer.QT_4bit,
    "QT_6bit":          faiss.ScalarQuantizer.QT_6bit,
    "6bit":             faiss.ScalarQuantizer.QT_6bit,
    "QT_fp16":          faiss.ScalarQuantizer.QT_fp16,
    "fp16":             faiss.ScalarQuantizer.QT_fp16,
    "QT_bf16":          faiss.ScalarQuantizer.QT_bf16,
    "bf16":             faiss.ScalarQuantizer.QT_bf16,
    "QT_8bit_uniform":  faiss.ScalarQuantizer.QT_8bit_uniform,
    "8bit_uniform":     faiss.ScalarQuantizer.QT_8bit_uniform,
    "QT_4bit_uniform":  faiss.ScalarQuantizer.QT_4bit_uniform,
    "4bit_uniform":     faiss.ScalarQuantizer.QT_4bit_uniform,
}


def resolve_qtype(qtype_str):
    qt = QTYPE_MAP.get(qtype_str)
    if qt is None:
        raise ValueError(f"Unknown qtype '{qtype_str}'. Valid: {list(QTYPE_MAP.keys())}")
    return qt


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS IndexHNSWSQ")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (must match section in datasets.conf)")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Config directory (default: ../config relative to script)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override dataset base path")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of threads")
    parser.add_argument("--ef", type=str, default=None,
                        help="Comma-separated ef_search values")
    args = parser.parse_args()

    # Resolve config directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = args.config_dir or os.path.join(script_dir, "..", "config")
    config_dir = os.path.abspath(config_dir)

    # Parse configs
    ds_configs = parse_datasets_conf(os.path.join(config_dir, "datasets.conf"))
    sq_configs = parse_sq_conf(os.path.join(config_dir, "sq.conf"))

    dataset = args.dataset
    ds_key = None
    for key in ds_configs:
        if key.lower() == dataset.lower():
            ds_key = key
            break
    if ds_key is None:
        print(f"Error: dataset '{dataset}' not found in datasets.conf")
        print(f"Available: {list(ds_configs.keys())}")
        return

    sq_key = None
    for key in sq_configs:
        if key.lower() == dataset.lower():
            sq_key = key
            break
    if sq_key is None:
        print(f"Error: dataset '{dataset}' not found in sq.conf")
        print(f"Available: {list(sq_configs.keys())}")
        return

    ds = ds_configs[ds_key]
    base_path = args.data_path or ds["base_path"]
    threads = args.threads or ds.get("threads", 16)
    k = ds.get("k", 10)
    hnsw_M = ds.get("hnsw_M", 64)
    hnsw_efc = ds.get("hnsw_efConstruction", 200)
    ef_values = ([int(x) for x in args.ef.split(",")] if args.ef
                 else ds.get("ef_search", [10, 20, 30, 50, 100, 200]))
    qtypes = sq_configs[sq_key]

    faiss.omp_set_num_threads(threads)

    print("=" * 50)
    print("IndexHNSWSQ Benchmark")
    print(f"Dataset:           {dataset}")
    print(f"Data path:         {base_path}")
    print(f"HNSW M:            {hnsw_M}")
    print(f"HNSW efConstruct:  {hnsw_efc}")
    print(f"Threads:           {threads}")
    print(f"SQ qtypes:         {qtypes}")
    print("=" * 50)

    # Load data
    print("\n[Loading data...]")
    t0 = time.time()
    xb = fvecs_read(os.path.join(base_path, ds["base_file"]))
    xq = fvecs_read(os.path.join(base_path, ds["query_file"]))
    gt = ivecs_read(os.path.join(base_path, ds["groundtruth_file"]))
    print(f"Load time: {time.time() - t0:.1f}s")

    nb, d = xb.shape
    nq = xq.shape[0]
    gt_k = gt.shape[1]
    print(f"Database: {nb} vectors, dimension {d}")
    print(f"Queries:  {nq} vectors")
    print(f"Ground truth: top-{gt_k}")

    # Run benchmarks for each SQ qtype
    for qtype_str in qtypes:
        try:
            qtype = resolve_qtype(qtype_str)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        print(f"\n{'=' * 50}")
        print(f"IndexHNSWSQ | qtype={qtype_str}")
        print(f"{'=' * 50}")

        # Build index
        print(f"[Building IndexHNSWSQ (qtype={qtype_str})...]")
        t0 = time.time()
        index = faiss.IndexHNSWSQ(d, qtype, hnsw_M)
        index.hnsw.efConstruction = hnsw_efc
        index.train(xb)
        index.add(xb)
        build_time = time.time() - t0
        print(f"Build time: {build_time:.1f}s")

        # Wrap with IndexRefine for exact-distance reranking
        flat_index = faiss.IndexFlatL2(d)
        flat_index.add(xb)
        refine_index = faiss.IndexRefine(index, flat_index)

        # Search sweep
        print(f"\n{'ef':>8s}{'QPS':>12s}{'Recall@' + str(k):>12s}"
              f"{'Latency(ms)':>12s}{'ndis_mean':>12s}{'nhops_mean':>12s}")
        print("-" * 68)

        for ef in ef_values:
            index.hnsw.efSearch = ef
            refine_index.k_factor = ef / k

            faiss.cvar.hnsw_stats.reset()

            t0 = time.time()
            D, I = refine_index.search(xq, k)
            search_time = time.time() - t0

            ndis_mean = faiss.cvar.hnsw_stats.ndis / nq
            nhops_mean = faiss.cvar.hnsw_stats.nhops / nq
            qps = nq / search_time
            latency_ms = search_time / nq * 1000
            recall = compute_recall(I, gt, k)

            print(f"{ef:>8d}{qps:>12.0f}{recall:>12.4f}"
                  f"{latency_ms:>12.3f}{ndis_mean:>12.1f}{nhops_mean:>12.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
