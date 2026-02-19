import os
import re
from dataclasses import dataclass


@dataclass
class ResultEntry:
    ef: int
    qps: float
    recall: float
    latency: float
    ndis: float | None = None
    nhops: float | None = None


@dataclass
class ResultFile:
    filepath: str
    entries: list        # list of ResultEntry
    build_time: float | None = None  # seconds, parsed from [Configuration] if present


def _detect_ndis_nhops_columns(header_parts: list) -> tuple:
    """From the column header tokens, find the indices for ndis and nhops.

    Returns (ndis_idx, nhops_idx) where each may be None.

    Column layouts seen:
      Standard:       ef  QPS  Recall  Latency(ms)
      Quant:          ef  QPS  Recall  Latency(ms)  ndis_mean   nhops_mean
      RaBitQ-Native:  ef  QPS  Recall  Latency(ms)  ndis1_mean  ndis2_mean  nhops_mean
    """
    ndis_idx = nhops_idx = None
    for i, col in enumerate(header_parts):
        col_l = col.lower()
        if ndis_idx is None and col_l.startswith("ndis"):
            ndis_idx = i
        if col_l.startswith("nhops"):
            nhops_idx = i
    return ndis_idx, nhops_idx


def parse_result_file(filepath: str) -> ResultFile:
    """Parse a benchmark result .txt file and extract summary rows.

    Supports both the standard format (comment headers) and
    the quantization format ([Summary] section).
    Extracts ef, QPS, Recall, Latency, and optionally ndis/nhops.
    """
    entries = []
    build_time = None
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_summary = False
    ndis_idx = nhops_idx = None

    for line in lines:
        stripped = line.strip()

        # Parse build time from [Configuration] section
        m = re.match(r"Build Time:\s+([\d.]+)\s*s", stripped)
        if m:
            build_time = float(m.group(1))

        # Detect quantization format [Summary] marker
        if stripped == "[Summary]":
            in_summary = True
            continue

        # Detect column header line (works for both standard and quant formats)
        if stripped.startswith("ef") and "QPS" in stripped and "Recall" in stripped:
            in_summary = True
            header_parts = stripped.split()
            ndis_idx, nhops_idx = _detect_ndis_nhops_columns(header_parts)
            continue

        # Skip separator lines
        if in_summary and stripped.startswith("---"):
            continue

        # End of summary table: blank line or next section
        if in_summary and (stripped == "" or stripped.startswith("[")):
            if entries:
                break
            continue

        if in_summary:
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    ef = int(parts[0])
                    qps = float(parts[1])
                    recall = float(parts[2])
                    # if (recall < 0.8):
                    #     continue
                    latency = float(parts[3])
                    ndis = float(parts[ndis_idx]) if ndis_idx is not None and ndis_idx < len(parts) else None
                    nhops = float(parts[nhops_idx]) if nhops_idx is not None and nhops_idx < len(parts) else None
                    entries.append(ResultEntry(ef=ef, qps=qps, recall=recall, latency=latency, ndis=ndis, nhops=nhops))
                except ValueError:
                    continue

    return ResultFile(filepath=filepath, entries=entries, build_time=build_time)


def load_all_results(results_dir: str, dataset: str, algo_folder: str) -> list:
    """Load all .txt result files for a given dataset/algorithm folder.

    Returns a list of ResultFile objects.
    """
    folder = os.path.join(results_dir, dataset, algo_folder)
    if not os.path.isdir(folder):
        return []

    result_files = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            rf = parse_result_file(os.path.join(folder, fname))
            if rf.entries:
                result_files.append(rf)
    return result_files


def select_best_config(result_files: list, target_recall: float) -> ResultFile | None:
    """Select the result file (configuration) that achieves the highest QPS
    at or above the target recall.

    For each file, we find the row with the smallest recall >= target_recall
    and record its QPS. The file with the highest such QPS wins.
    If no file reaches the target recall, return the one whose max recall is highest.
    """
    best_file = None
    best_qps = -1.0

    # First pass: files that reach target recall
    for rf in result_files:
        candidates = [e for e in rf.entries if e.recall >= target_recall]
        if candidates:
            # Among rows that meet target, pick the one with highest QPS
            # (lower ef -> higher QPS, but let's just pick max QPS directly)
            top = max(candidates, key=lambda e: e.qps)
            if top.qps > best_qps:
                best_qps = top.qps
                best_file = rf

    # Fallback: no file reaches target recall, pick one with highest max recall
    if best_file is None:
        best_recall = -1.0
        for rf in result_files:
            max_recall = max(e.recall for e in rf.entries)
            if max_recall > best_recall:
                best_recall = max_recall
                best_file = rf

    return best_file


# ── Label generation ──────────────────────────────────────────

_METHOD_NAMES = {
    "standard": "Standard",
    "pq": "PQ",
    "sq": "SQ",
    "rabitq": "RaBitQ",
    "rabitq_native": "RaBitQ-Native",
    "opq": "OPQ",
    "vaq": "VAQ",
    "rq": "RQ",
    "prq": "PQR",
    "lsq": "LSQ",
    "plsq": "PLSQ",
}


def make_label(relpath: str) -> str:
    """Generate a legend label from a relative result path.

    E.g. "pq/pq_M16_nbits8_M32_efc200.txt" -> "PQ (M16, nbits8)"
         "rabitq_native/rabitq_native_bits1_c128_rerank_M32_efc200.txt"
           -> "RaBitQ-Native (bits1, c128, rerank)"
    """
    folder = relpath.split("/")[0]
    fname = os.path.splitext(os.path.basename(relpath))[0]

    method = _METHOD_NAMES.get(folder, folder)

    # Strip algo prefix
    prefix = folder + "_"
    if fname.startswith(prefix):
        rest = fname[len(prefix):]
    else:
        rest = fname

    # Strip HNSW suffix (_M{d}_efc{d})
    rest = re.sub(r"_M\d+_efc\d+$", "", rest)

    # Strip "rerank" from rabitq_native params
    if folder == "rabitq_native":
        rest = re.sub(r"_?rerank_?", "", rest).strip("_")

    if rest:
        params = rest.replace("_", ", ")
        return f"{method} ({params})"
    return method