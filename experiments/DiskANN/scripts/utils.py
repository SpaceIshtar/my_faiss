import os
from dataclasses import dataclass


@dataclass
class ResultEntry:
    search_param: int
    qps: float
    recall: float
    latency: float
    ndis: float | None = None
    nhops: float | None = None


@dataclass
class ResultFile:
    filepath: str
    entries: list[ResultEntry]


def _detect_ndis_nhops_columns(header_parts: list[str]) -> tuple[int | None, int | None]:
    ndis_idx = None
    nhops_idx = None
    for idx, col in enumerate(header_parts):
        lower = col.lower()
        if ndis_idx is None and lower.startswith("ndis"):
            ndis_idx = idx
        if nhops_idx is None and lower.startswith("nhops"):
            nhops_idx = idx
    return ndis_idx, nhops_idx


def parse_result_file(filepath: str) -> ResultFile:
    entries: list[ResultEntry] = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_summary = False
    ndis_idx = None
    nhops_idx = None

    for line in lines:
        stripped = line.strip()

        if stripped == "[Summary]":
            in_summary = True
            continue

        if stripped.startswith("L") and "QPS" in stripped and "Recall" in stripped:
            in_summary = True
            header_parts = stripped.split()
            ndis_idx, nhops_idx = _detect_ndis_nhops_columns(header_parts)
            continue

        if in_summary and stripped.startswith("---"):
            continue

        if in_summary and (not stripped or stripped.startswith("[")):
            if entries:
                break
            continue

        if not in_summary:
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue

        try:
            search_param = int(parts[0])
            qps = float(parts[1])
            recall = float(parts[2])
            latency = float(parts[3])
            ndis = float(parts[ndis_idx]) if ndis_idx is not None and ndis_idx < len(parts) else None
            nhops = float(parts[nhops_idx]) if nhops_idx is not None and nhops_idx < len(parts) else None
        except ValueError:
            continue

        entries.append(
            ResultEntry(
                search_param=search_param,
                qps=qps,
                recall=recall,
                latency=latency,
                ndis=ndis,
                nhops=nhops,
            )
        )

    return ResultFile(filepath=filepath, entries=entries)


def load_all_results(results_dir: str, dataset: str, algo_folder: str) -> list[ResultFile]:
    folder = os.path.join(results_dir, dataset, algo_folder)
    if not os.path.isdir(folder):
        return []

    result_files: list[ResultFile] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        rf = parse_result_file(os.path.join(folder, fname))
        if rf.entries:
            result_files.append(rf)
    return result_files


def select_best_config(result_files: list[ResultFile], target_recall: float) -> ResultFile | None:
    best_file = None
    best_qps = -1.0

    for rf in result_files:
        candidates = [entry for entry in rf.entries if entry.recall >= target_recall]
        if not candidates:
            continue
        top = max(candidates, key=lambda entry: entry.qps)
        if top.qps > best_qps:
            best_qps = top.qps
            best_file = rf

    if best_file is not None:
        return best_file

    best_recall = -1.0
    for rf in result_files:
        max_recall = max(entry.recall for entry in rf.entries)
        if max_recall > best_recall:
            best_recall = max_recall
            best_file = rf

    return best_file
