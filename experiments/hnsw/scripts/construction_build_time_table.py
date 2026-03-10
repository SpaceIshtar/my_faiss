import argparse
from pathlib import Path

from utils import parse_result_file


MODES = ("exact_sdc", "adc_exact", "adc_sdc")
PQ_OPQ_ROWS = (
    ("M120", "nbits4"),
    ("M240", "nbits4"),
    ("M480", "nbits4"),
    ("M120", "nbits8"),
    ("M240", "nbits8"),
    ("M480", "nbits8"),
)
SQ_ROWS = ("4bit", "6bit", "8bit")


def _format_time(v: float | None) -> str:
    if v is None:
        return "todo"
    return f"{v:.1f}"


def _read_build_time(path: Path) -> float | None:
    if not path.is_file():
        return None
    rf = parse_result_file(str(path))
    if rf.build_time is None or rf.build_time <= 0:
        return None
    return rf.build_time


def _build_time_by_mode(base_dir: Path, file_stem_prefix: str) -> list[str]:
    vals: list[str] = []
    for mode in MODES:
        fpath = base_dir / f"{file_stem_prefix}_{mode}_M64_efc200.txt"
        vals.append(_format_time(_read_build_time(fpath)))
    return vals


def _emit_block(name: str, rows: list[tuple[str, list[str]]]) -> None:
    print(r"\midrule")
    print(
        rf"        \multirow{{{len(rows)}}}{{4em}}{{{name}}} "
        rf"& {rows[0][0]} & {' & '.join(rows[0][1])}\\"
    )
    for row_name, vals in rows[1:]:
        print(rf"                              & {row_name} & {' & '.join(vals)}\\")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LaTeX table rows from HNSW construction Build Time logs."
    )
    parser.add_argument(
        "--dataset",
        default="gist1M",
        help="Dataset name under experiments/hnsw/results (default: gist1M).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    construction_dir = script_dir.parent / "results" / args.dataset / "construction"

    pq_rows: list[tuple[str, list[str]]] = []
    for m, nbits in PQ_OPQ_ROWS:
        label = f"{m},{nbits}"
        prefix = f"pq_{m}_{nbits}"
        pq_rows.append((label, _build_time_by_mode(construction_dir / "pq", prefix)))
    _emit_block("PQ", pq_rows)

    opq_rows: list[tuple[str, list[str]]] = []
    for m, nbits in PQ_OPQ_ROWS:
        label = f"{m},{nbits}"
        prefix = f"opq_{m}_{nbits}"
        opq_rows.append((label, _build_time_by_mode(construction_dir / "opq", prefix)))
    _emit_block("OPQ", opq_rows)

    sq_rows: list[tuple[str, list[str]]] = []
    for bits in SQ_ROWS:
        label = f"nbits{bits[0]}"
        prefix = f"sq_QT_{bits}"
        sq_rows.append((label, _build_time_by_mode(construction_dir / "sq", prefix)))
    _emit_block("SQ", sq_rows)


if __name__ == "__main__":
    main()
