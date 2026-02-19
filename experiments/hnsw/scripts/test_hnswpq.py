"""
Benchmark FAISS IndexHNSWPQ with exact-distance reranking.

This script builds an IndexHNSWPQ (HNSW graph with PQ storage) and wraps it
with IndexRefineFlat. For each ef_search value, HNSW traverses the graph using
PQ distances to collect ef candidates, then reranks them with exact L2
distances and returns the top-k results.

Usage:
    python test_hnswpq.py --dataset sift1m
    python test_hnswpq.py --dataset gist1M --threads 8
    python test_hnswpq.py --dataset sift1m --data-path /alt/path
"""

import argparse
import configparser
import os
import struct
import time

import faiss
import numpy as np


# ── Config parsing ────────────────────────────────────────────

def parse_datasets_conf(filepath):
    """Parse datasets.conf (INI-style) into a dict of dataset configs."""
    cp = configparser.ConfigParser()
    cp.read(filepath)
    datasets = {}
    for section in cp.sections():
        cfg = dict(cp[section])
        # Convert numeric fields
        for int_key in ("threads", "k", "hnsw_M", "hnsw_efConstruction"):
            if int_key in cfg:
                cfg[int_key] = int(cfg[int_key])
        if "ef_search" in cfg:
            cfg["ef_search"] = [int(x.strip()) for x in cfg["ef_search"].split(",")]
        datasets[section] = cfg
    return datasets


def parse_pq_conf(filepath):
    """Parse pq.conf into a dict mapping dataset -> list of {M, nbits}."""
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
            elif current_section is not None:
                params = {}
                for token in line.split(","):
                    token = token.strip()
                    if "=" in token:
                        k, v = token.split("=", 1)
                        params[k.strip()] = int(v.strip())
                if params:
                    param_sets[current_section].append(params)

    return param_sets


# ── Data I/O (fvecs / ivecs) ─────────────────────────────────

def fvecs_read(filepath):
    """Read .fvecs file, return numpy float32 array of shape (n, d)."""
    with open(filepath, "rb") as f:
        buf = f.read()

    d = struct.unpack_from("<i", buf, 0)[0]
    row_bytes = 4 + d * 4  # 4 bytes for dim header + d floats
    n = len(buf) // row_bytes

    data = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        offset = i * row_bytes + 4  # skip the dim header
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


# ── Result output ─────────────────────────────────────────────

def save_results(filepath, dataset_name, pq_M, pq_nbits,
                 hnsw_M, hnsw_efc, nq, k, results):
    """Save benchmark results to a text file matching the C++ output format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("========================================\n")
        f.write("HNSW + PQ (IndexHNSWPQ) Benchmark Results\n")
        f.write("========================================\n\n")

        f.write("[Configuration]\n")
        f.write(f"  Dataset:           {dataset_name}\n")
        f.write(f"  Algorithm:         IndexHNSWPQ\n")
        f.write(f"  PQ M:              {pq_M}\n")
        f.write(f"  PQ nbits:          {pq_nbits}\n")
        f.write(f"  HNSW M:            {hnsw_M}\n")
        f.write(f"  HNSW efConstruct:  {hnsw_efc}\n")
        f.write(f"  Num Queries:       {nq}\n")
        f.write(f"  Recall@k:          {k}\n\n")

        f.write("[Summary]\n")
        f.write(f"{'ef':>8s}{'QPS':>12s}{'Recall':>12s}"
                f"{'Latency(ms)':>12s}{'ndis_mean':>12s}{'nhops_mean':>12s}\n")
        f.write("-" * 68 + "\n")

        for r in results:
            f.write(f"{r['ef']:>8d}{r['qps']:>12.0f}"
                    f"{r['recall']:>12.4f}{r['latency_ms']:>12.3f}"
                    f"{r['ndis_mean']:>12.1f}{r['nhops_mean']:>12.1f}\n")

        f.write("\n========================================\n")
        f.write("End of Results\n")
        f.write("========================================\n")

    print(f"Results saved to: {filepath}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS IndexHNSWPQ")
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
    pq_configs = parse_pq_conf(os.path.join(config_dir, "pq.conf"))

    dataset = args.dataset
    # Case-insensitive lookup
    ds_key = None
    for key in ds_configs:
        if key.lower() == dataset.lower():
            ds_key = key
            break
    if ds_key is None:
        print(f"Error: dataset '{dataset}' not found in datasets.conf")
        print(f"Available: {list(ds_configs.keys())}")
        return

    ds = ds_configs[ds_key]
    base_path = args.data_path or ds["base_path"]
    threads = args.threads or ds.get("threads", 16)
    k = ds.get("k", 10)
    hnsw_M = ds.get("hnsw_M", 32)
    hnsw_efc = ds.get("hnsw_efConstruction", 200)
    ef_values = ([int(x) for x in args.ef.split(",")] if args.ef
                 else ds.get("ef_search", [10, 20, 30, 50, 100, 200]))

    faiss.omp_set_num_threads(threads)

    # PQ param sets for this dataset
    pq_key = None
    for key in pq_configs:
        if key.lower() == dataset.lower():
            pq_key = key
            break
    if pq_key is None:
        print(f"Error: dataset '{dataset}' not found in pq.conf")
        print(f"Available: {list(pq_configs.keys())}")
        return

    pq_param_sets = pq_configs[pq_key]

    print("=" * 50)
    print("IndexHNSWPQ Benchmark")
    print(f"Dataset:           {dataset}")
    print(f"Data path:         {base_path}")
    print(f"HNSW M:            {hnsw_M}")
    print(f"HNSW efConstruct:  {hnsw_efc}")
    print(f"Threads:           {threads}")
    print(f"PQ configs:        {len(pq_param_sets)} parameter sets")
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

    # Result output directory
    results_dir = os.path.join(script_dir, "..", "results", dataset, "hnswpq")

    # Run benchmarks for each PQ parameter set
    for ps in pq_param_sets:
        pq_M = ps["M"]
        pq_nbits = ps.get("nbits", 8)

        print(f"\n{'=' * 50}")
        print(f"IndexHNSWPQ | PQ M={pq_M}, nbits={pq_nbits}")
        print(f"{'=' * 50}")

        # Check dimension divisibility
        if d % pq_M != 0:
            print(f"[SKIP] dimension {d} not divisible by PQ M={pq_M}")
            continue

        # Build or load index
        index_dir = os.path.join(base_path, "index")
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(
            index_dir,
            f"hnswpq_M{pq_M}_nbits{pq_nbits}_hM{hnsw_M}_efc{hnsw_efc}.faissindex")

        if os.path.exists(index_path):
            print(f"[Loading index from {index_path}...]")
            t0 = time.time()
            index = faiss.read_index(index_path)
            print(f"Load time: {time.time() - t0:.1f}s")
        else:
            print(f"[Building IndexHNSWPQ (PQ M={pq_M}, nbits={pq_nbits})...]")
            t0 = time.time()
            index = faiss.IndexHNSWPQ(d, pq_M, hnsw_M, pq_nbits)
            index.hnsw.efConstruction = hnsw_efc
            index.train(xb)
            index.add(xb)
            build_time = time.time() - t0
            print(f"Build time: {build_time:.1f}s")

            print(f"[Saving index to {index_path}...]")
            faiss.write_index(index, index_path)

        # Wrap with IndexRefine for exact-distance reranking
        # Build the flat index first so ntotal matches the base
        flat_index = faiss.IndexFlatL2(d)
        flat_index.add(xb)
        refine_index = faiss.IndexRefine(index, flat_index)

        # Search benchmark: for each ef, HNSW explores ef candidates using
        # PQ distances, retrieves ef results, then reranks with exact distances
        print(f"\n{'ef':>8s}{'QPS':>12s}{'Recall@' + str(k):>12s}"
              f"{'Latency(ms)':>12s}{'ndis_mean':>12s}{'nhops_mean':>12s}")
        print("-" * 68)

        results = []
        for ef in ef_values:
            index.hnsw.efSearch = ef
            refine_index.k_factor = ef / k  # retrieve ef candidates, rerank to k

            # Reset HNSW stats before search
            faiss.cvar.hnsw_stats.reset()

            # Timed search
            t0 = time.time()
            D, I = refine_index.search(xq, k)
            search_time = time.time() - t0

            # Read accumulated stats
            ndis_mean = faiss.cvar.hnsw_stats.ndis / nq
            nhops_mean = faiss.cvar.hnsw_stats.nhops / nq

            qps = nq / search_time
            latency_ms = search_time / nq * 1000
            recall = compute_recall(I, gt, k)

            results.append({
                "ef": ef,
                "qps": qps,
                "recall": recall,
                "latency_ms": latency_ms,
                "ndis_mean": ndis_mean,
                "nhops_mean": nhops_mean,
            })

            print(f"{ef:>8d}{qps:>12.0f}{recall:>12.4f}"
                  f"{latency_ms:>12.3f}{ndis_mean:>12.1f}{nhops_mean:>12.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
