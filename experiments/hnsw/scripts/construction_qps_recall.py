import os
import re
import matplotlib.pyplot as plt
import matplotlib
from utils import parse_result_file, select_best_config

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ── Configuration ──────────────────────────────────────────────
DATASET = "gist1M"
METHOD = "sq"
TARGET_RECALL = 0.90
TARGET_TIME = 361.467
MIN_EF = 20  # skip ef values below this threshold in all plots
NAME = f"{DATASET}_construction_{METHOD}"

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results", DATASET)
CONSTRUCTION_DIR = os.path.join(RESULTS_DIR, "construction", METHOD)
STANDARD_DIR = os.path.join(RESULTS_DIR, "standard")
FIGURES_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", "hnsw", DATASET)
LEGENDS_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", "hnsw", "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Mode suffixes (search_mode + pruning_mode) ────────────────
_MODES = ["adc_exact", "adc_sdc", "exact_sdc"]

_MODE_STYLE = {
    "exact_exact": {"label": "Exact + Exact", "color": "#000000", "marker": "o"},
    "exact_sdc":   {"label": "Exact + SDC",   "color": "#d62728", "marker": "s"},
    "adc_exact":   {"label": "ADC + Exact",   "color": "#2ca02c", "marker": "^"},
    "adc_sdc":     {"label": "ADC + SDC",     "color": "#1f77b4", "marker": "D"},
}


def extract_params(fname: str, method: str) -> tuple[str, str] | None:
    """Extract (quant_params, mode) from a construction filename.

    E.g. "pq_M120_nbits4_adc_exact_M64_efc200.txt"
         -> ("M120_nbits4", "adc_exact")
    """
    stem = os.path.splitext(fname)[0]  # strip .txt
    prefix = method + "_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix):]           # M120_nbits4_adc_exact_M64_efc200
    rest = re.sub(r"_M\d+_efc\d+$", "", rest)  # M120_nbits4_adc_exact
    for mode in _MODES:
        if rest.endswith("_" + mode):
            params = rest[: -(len(mode) + 1)]
            return params, mode
    return None


# ── Find best params across ALL construction files ─────────────
all_files = [
    f for f in os.listdir(CONSTRUCTION_DIR)
    if f.endswith(".txt")
]

best_qps = -1.0
best_params = None
# fallback: file with minimum build time (used when no file passes both filters)
fallback_params = None
fallback_time = float("inf")

for fname in sorted(all_files):
    info = extract_params(fname, METHOD)
    if info is None:
        continue
    params, mode = info
    rf = parse_result_file(os.path.join(CONSTRUCTION_DIR, fname))

    # Track global minimum build time as fallback
    if rf.build_time is not None and rf.build_time < fallback_time:
        fallback_time = rf.build_time
        fallback_params = params

    if TARGET_TIME is not None and rf.build_time is not None and rf.build_time > TARGET_TIME:
        continue
    candidates = [e for e in rf.entries if e.recall >= TARGET_RECALL]
    if candidates:
        top = max(candidates, key=lambda e: e.qps)
        if top.qps > best_qps:
            best_qps = top.qps
            best_params = params
            print(f"  New best: {fname}  (recall={top.recall:.4f}, qps={top.qps:.0f})")

if best_params is None and fallback_params is not None:
    best_params = fallback_params
    print(f"  No file passed both filters; falling back to min build_time: {fallback_params} ({fallback_time:.1f}s)")

if best_params is None:
    print("No construction file reached target recall. Exiting.")
    exit(1)

print(f"\nSelected params: {best_params}")

# ── Load the 3 construction curves for best_params ───────────
curves = []

for mode in _MODES:
    fname = f"{METHOD}_{best_params}_{mode}_M64_efc200.txt"
    fpath = os.path.join(CONSTRUCTION_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"[SKIP] Not found: {fname}")
        continue
    rf = parse_result_file(fpath)
    if not rf.entries:
        print(f"[SKIP] No data: {fname}")
        continue
    style = _MODE_STYLE[mode]
    curves.append((style["label"], sorted([e for e in rf.entries if e.ef >= MIN_EF], key=lambda e: e.recall),
                   style["color"], style["marker"]))
    print(f"[OK]   {style['label']}: {fname}")

# ── Load Standard (exact+exact) ───────────────────────────────
std_files = [f for f in os.listdir(STANDARD_DIR) if f.endswith(".txt")]
if std_files:
    std_rf = select_best_config(
        [parse_result_file(os.path.join(STANDARD_DIR, f)) for f in std_files],
        TARGET_RECALL,
    )
    if std_rf and std_rf.entries:
        style = _MODE_STYLE["exact_exact"]
        curves.insert(0, (style["label"],
                          sorted(std_rf.entries, key=lambda e: e.recall),
                          style["color"], style["marker"]))
        print(f"[OK]   {style['label']}: {os.path.basename(std_rf.filepath)}")

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
handles = []

for label, entries, color, marker in curves:
    recalls = [e.recall for e in entries]
    qps_vals = [e.qps for e in entries]
    line, = ax.plot(
        recalls, qps_vals,
        marker=marker, color=color, linestyle="-",
        linewidth=2, markersize=6, label=label,
    )
    handles.append(line)

ax.set_xlabel("Recall@10")
ax.set_ylabel("QPS")
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
# ax.set_title(f"{DATASET} {METHOD.upper()} ({best_params}) — Construction Mode")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"{NAME}_qps_recall.pdf"), bbox_inches="tight")
print(f"\nSaved: {NAME}_qps_recall.pdf")

# ── Legend ────────────────────────────────────────────────────
if handles:
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(os.path.join(LEGENDS_DIR, f"{NAME}_legend.pdf"),
                    bbox_inches=bbox, pad_inches=0)
    print(f"Saved legend: {NAME}_legend.pdf")

plt.close("all")
