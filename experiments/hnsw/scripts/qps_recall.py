import os
import matplotlib.pyplot as plt
import matplotlib
from utils import load_all_results, select_best_config

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ── Configuration ──────────────────────────────────────────────
DATASET = "gist1M"
ALGORITHMS = {
    "Standard": "standard",
    "PQ": "pq",
    "SQ": "sq",
    "RaBitQ": "rabitq",
    "RaBitQ-Native": "rabitq_native",
    "OPQ": "opq",
    "VAQ": "vaq",
    "RQ": "rq",
    "PQR": "prq",
    "LSQ": "lsq",
    "PLSQ": "plsq",
}
TARGET_RECALL = 0.95
NAME = "gist1M_qps_recall_0.95.pdf"

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", DATASET)
LEGENDS_DIR = os.path.join("/common/home/jl3288/overleaf/quantization_benchmark", "figures", "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Style definitions ─────────────────────────────────────────
STYLES = {
    "Standard":     {"color": "#000000", "marker": "o",  "linestyle": "-"},
    "PQ":           {"color": "#1f77b4", "marker": "s",  "linestyle": "-"},
    "SQ":           {"color": "#ff7f0e", "marker": "^",  "linestyle": "-"},
    "RaBitQ":       {"color": "#2ca02c", "marker": "D",  "linestyle": "-"},
    "RaBitQ-Native":{"color": "#d62728", "marker": "v",  "linestyle": "-"},
    "OPQ":          {"color": "#9467bd", "marker": "P",  "linestyle": "-"},
    "VAQ":          {"color": "#8c564b", "marker": "X",  "linestyle": "-"},
    "RQ":           {"color": "#e377c2", "marker": "h",  "linestyle": "-"},
    "PQR":          {"color": "#7f7f7f", "marker": "*",  "linestyle": "-"},
    "LSQ":          {"color": "#bcbd22", "marker": "p",  "linestyle": "-"},
    "PLSQ":         {"color": "#17becf", "marker": "d",  "linestyle": "-"},
}

# ── Collect data ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
legend_handles = []

for algo_name, algo_folder in ALGORITHMS.items():
    result_files = load_all_results(RESULTS_DIR, DATASET, algo_folder)
    if not result_files:
        print(f"[SKIP] {algo_name}: no result files found in {DATASET}/{algo_folder}")
        continue

    best = select_best_config(result_files, TARGET_RECALL)
    if best is None:
        print(f"[SKIP] {algo_name}: could not select a config")
        continue

    entries = sorted(best.entries, key=lambda e: e.recall)
    recalls = [e.recall for e in entries]
    qps_vals = [e.qps for e in entries]

    style = STYLES.get(algo_name, {"color": "gray", "marker": ".", "linestyle": "-"})
    line, = ax.plot(
        recalls, qps_vals,
        marker=style["marker"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=2,
        markersize=6,
        label=algo_name,
    )
    legend_handles.append(line)

    config_name = os.path.basename(best.filepath)
    print(f"[OK]   {algo_name}: selected {config_name}")

# ── Format axes ───────────────────────────────────────────────
ax.set_xlabel("Recall@10")
ax.set_ylabel("QPS")
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
# ax.set_title(f"{DATASET} — QPS vs Recall")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, NAME), format='pdf', bbox_inches="tight")
print(f"\nFigure saved to {os.path.join(FIGURES_DIR, NAME)}")

# ── Save legend separately ────────────────────────────────────
# if legend_handles:
#     fig_leg = plt.figure()
#     legend = fig_leg.legend(
#         handles=legend_handles,
#         loc="center",
#         ncol=6,
#         frameon=False,
#         fontsize=12,
#     )
#     fig_leg.canvas.draw()
#     bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
#     legend_path = os.path.join(LEGENDS_DIR, NAME)
#     fig_leg.savefig(legend_path, bbox_inches=bbox, pad_inches=0)
#     print(f"Legend saved to {legend_path}")

plt.close("all")