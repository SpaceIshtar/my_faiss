import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
from utils import parse_result_file, make_label

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ── Configuration ──────────────────────────────────────────────
DATASET = "video"
FILES = [
    "standard/standard_M64_efc200.txt",
    "rabitq/rabitq_bits1_c16_M64_efc200.txt",
    "rabitq/rabitq_bits1_c512_M64_efc200.txt",
    # "rabitq/rabitq_bits1_c4096_M64_efc200.txt",
    "rabitq/rabitq_bits2_c16_M64_efc200.txt",
    "rabitq/rabitq_bits2_c512_M64_efc200.txt",
    "rabitq/rabitq_bits2_c4096_M64_efc200.txt",
    "rabitq/rabitq_bits4_c16_M64_efc200.txt",
    "rabitq/rabitq_bits4_c512_M64_efc200.txt",
    # "rabitq/rabitq_bits4_c4096_M64_efc200.txt",
    # "rabitq/rabitq_bits8_c16_M64_efc200.txt",
    # "rabitq/rabitq_bits8_c512_M64_efc200.txt",
    # "rabitq/rabitq_bits8_c4096_M64_efc200.txt",
    "rabitq_native/rabitq_native_bits2_c16_rerank_M64_efc200.txt",
    "rabitq_native/rabitq_native_bits2_c512_rerank_M64_efc200.txt",
    "rabitq_native/rabitq_native_bits4_c16_rerank_M64_efc200.txt",
    "rabitq_native/rabitq_native_bits4_c512_rerank_M64_efc200.txt",
    # "rabitq_native/rabitq_native_bits4_c4096_rerank_M64_efc200.txt",
    # "rabitq_native/rabitq_native_bits8_c16_rerank_M64_efc200.txt",
    # "rabitq_native/rabitq_native_bits8_c512_rerank_M64_efc200.txt",
    # "rabitq_native/rabitq_native_bits8_c4096_rerank_M64_efc200.txt",
]
NAME = "video_compare_rabitq"

# DATASET = "gist1M"
# FILES = [
#     "standard/standard_M64_efc200.txt",
#     "pq/pq_M480_nbits8_M64_efc200.txt",
#     "pq/pq_M240_nbits8_M64_efc200.txt",
#     "pq/pq_M120_nbits8_M64_efc200.txt",
#     "pq/pq_M480_nbits4_M64_efc200.txt",
#     "pq/pq_M240_nbits4_M64_efc200.txt",
#     # "pq/pq_M120_nbits4_M64_efc200.txt",
#     "opq/opq_M480_nbits8_niter50_M64_efc200.txt",
#     "opq/opq_M240_nbits8_niter50_M64_efc200.txt",
#     "opq/opq_M120_nbits8_niter50_M64_efc200.txt",
#     "opq/opq_M480_nbits4_niter50_M64_efc200.txt",
#     "opq/opq_M240_nbits4_niter50_M64_efc200.txt",
#     # "opq/opq_M120_nbits4_niter50_M64_efc200.txt",
# ]
# NAME = "gist1M_compare_pq"

MAX_EF = 100  # ef upper limit for ndis/nhops plots (None = no limit)

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results", DATASET)
FIGURES_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", "hnsw", DATASET)
LEGENDS_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", "hnsw", "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Style pool ─────────────────────────────────────────────────
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]
_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "h", "*", "p",
    "d", ">", "<", "H", "8", "1", "2", "3", "4", "+",
]

# ── Load data ──────────────────────────────────────────────────
color_iter = cycle(_COLORS)
marker_iter = cycle(_MARKERS)

curves = []  # list of (label, entries, color, marker)

for relpath in FILES:
    filepath = os.path.join(RESULTS_DIR, relpath)
    if not os.path.isfile(filepath):
        print(f"[SKIP] File not found: {relpath}")
        continue

    rf = parse_result_file(filepath)
    if not rf.entries:
        print(f"[SKIP] No data in: {relpath}")
        continue

    label = make_label(relpath)
    folder = relpath.split("/")[0]
    if folder == "standard":
        color = "#000000"
        marker = "o"
    else:
        color = next(color_iter)
        marker = next(marker_iter)
    entries = sorted(rf.entries, key=lambda e: e.ef)
    curves.append((label, entries, color, marker))
    print(f"[OK]   {label}")

if not curves:
    print("No data to plot.")
    exit(1)

# ── Helper ─────────────────────────────────────────────────────
def plot_curves(ax, curves, xkey, ykey, xlabel, ylabel, yscale="linear", xmax=None):
    handles = []
    for label, entries, color, marker in curves:
        if xmax is not None:
            entries = [e for e in entries if getattr(e, xkey) <= xmax]
        xs = [getattr(e, xkey) for e in entries]
        ys = [getattr(e, ykey) for e in entries]
        if any(v is None for v in ys):
            continue
        line, = ax.plot(
            xs, ys,
            marker=marker, color=color, linestyle="-",
            linewidth=2, markersize=6, label=label,
        )
        handles.append(line)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    return handles

# ── Figure 1: QPS vs Recall ──────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6))
handles = plot_curves(
    ax1, curves,
    xkey="recall", ykey="qps",
    xlabel="Recall@10", ylabel="QPS",
    yscale="log",
)
# ax1.set_title(f"{DATASET} — QPS vs Recall")
fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, f"{NAME}_qps_recall.pdf"), bbox_inches="tight")
print(f"\nSaved: {NAME}_qps_recall.pdf")

# ── Figure 2: ndis vs ef ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 6))
plot_curves(
    ax2, curves,
    xkey="ef", ykey="ndis",
    xlabel="ef", ylabel="# Distance Computations (ndis)",
    xmax=MAX_EF,
)
# ax2.set_title(f"{DATASET} — ndis vs ef")
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, f"{NAME}_ndis.pdf"), bbox_inches="tight")
print(f"Saved: {NAME}_ndis.pdf")

# ── Figure 3: nhops vs ef ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))
plot_curves(
    ax3, curves,
    xkey="ef", ykey="nhops",
    xlabel="ef", ylabel="# Graph Hops (nhops)",
    xmax=MAX_EF,
)
# ax3.set_title(f"nhops vs ef")
fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, f"{NAME}_nhops.pdf"), bbox_inches="tight")
print(f"Saved: {NAME}_nhops.pdf")

# ── Legend (separate) ─────────────────────────────────────────
if handles:
    fig_leg = plt.figure()
    ncol = min(len(handles), 2)
    legend = fig_leg.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=18,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(os.path.join(LEGENDS_DIR, f"{NAME}_legend.pdf"), bbox_inches=bbox, pad_inches=0)
    print(f"Saved legend: {NAME}_legend.pdf")

plt.close("all")
