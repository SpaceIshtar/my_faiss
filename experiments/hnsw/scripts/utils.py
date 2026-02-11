import os
import re
from dataclasses import dataclass


@dataclass
class ResultEntry:
    ef: int
    qps: float
    recall: float
    latency: float


@dataclass
class ResultFile:
    filepath: str
    entries: list  # list of ResultEntry


def parse_result_file(filepath: str) -> ResultFile:
    """Parse a benchmark result .txt file and extract (ef, QPS, Recall, Latency) rows.

    Supports both the standard format (comment headers) and
    the quantization format ([Summary] section).
    """
    entries = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_summary = False
    is_standard = False

    for line in lines:
        stripped = line.strip()

        # Detect standard format header line
        if stripped.startswith("ef") and "QPS" in stripped and "Recall" in stripped:
            in_summary = True
            is_standard = True
            continue

        # Detect quantization format [Summary] marker
        if stripped == "[Summary]":
            in_summary = True
            continue

        # Skip separator / column header lines in quantization format
        if in_summary and (stripped.startswith("---") or stripped.startswith("ef")):
            continue

        # End of summary table: blank line or next section
        if in_summary and (stripped == "" or stripped.startswith("[")):
            if entries:  # already collected data, stop
                break
            continue

        if in_summary:
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    ef = int(parts[0])
                    qps = float(parts[1])
                    recall = float(parts[2])
                    latency = float(parts[3])
                    entries.append(ResultEntry(ef=ef, qps=qps, recall=recall, latency=latency))
                except ValueError:
                    continue

    return ResultFile(filepath=filepath, entries=entries)


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