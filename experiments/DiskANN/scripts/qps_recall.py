import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from utils import load_all_results, select_best_config


matplotlib.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


ALGORITHMS = {
    "PQ": "pq",
    "SQ": "sq",
    "RaBitQ": "rabitq",
    "OPQ": "opq",
    "OSQ": "osq",
    "SAQ": "saq",
    "TurboQuant": "turboquant",
}

STYLES = {
    "PQ": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "SQ": {"color": "#ff7f0e", "marker": "^", "linestyle": "-"},
    "RaBitQ": {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    "OPQ": {"color": "#9467bd", "marker": "P", "linestyle": "-"},
    "OSQ": {"color": "#8c564b", "marker": "X", "linestyle": "-"},
    "SAQ": {"color": "#f39c12", "marker": "8", "linestyle": "-"},
    "TurboQuant": {"color": "#17becf", "marker": "o", "linestyle": "-"},
}

# Hardcode the search parameters you want to plot for each algorithm here.
# Use None to keep all points from the selected config.
# DiskANN result files use L as the search parameter, which plays the role
# of the search-time knob similarly to ef in HNSW.
PLOT_SEARCH_PARAMS = {
    "PQ": [20,30,50,70,100,150,200,300,400],
    "SQ": [20,30,50,70,100,150,200,300,400],
    "RaBitQ": [20,30,50,70,100,150,200,300,400],
    "OPQ": [20,30,50,70,100,150,200,300,400],
    "OSQ": [20,30,50,70,100,150,200,300,400],
    "SAQ": [20,30,50,70,100,150,200,300,400],
    "TurboQuant": [20,30,50,70,100,150,200,300,400],
}


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, "..", "results")
    default_figures_root = os.path.join(script_dir, "..", "figures")

    parser = argparse.ArgumentParser(description="Plot DiskANN QPS-recall curves.")
    parser.add_argument("--dataset", default="bigann", help="Dataset name under experiments/DiskANN/results.")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Recall threshold for selecting the best config.")
    parser.add_argument("--results-dir", default=default_results_dir, help="Root directory containing DiskANN result folders.")
    parser.add_argument("--figures-dir", default=default_figures_root, help="Root directory where figures and legends will be saved.")
    parser.add_argument("--output-name", default=None, help="Optional output filename. Defaults to <dataset>_qps_recall_<target>.pdf")
    return parser.parse_args()


def filter_entries_for_plot(algo_name: str, entries: list):
    selected_params = PLOT_SEARCH_PARAMS.get(algo_name)
    if selected_params is None:
        return entries

    selected_param_set = set(selected_params)
    return [entry for entry in entries if entry.search_param in selected_param_set]


def main() -> None:
    args = parse_args()

    dataset_figures_dir = os.path.join(args.figures_dir, args.dataset)
    legends_dir = os.path.join(args.figures_dir, "legends")
    os.makedirs(dataset_figures_dir, exist_ok=True)
    os.makedirs(legends_dir, exist_ok=True)

    if args.output_name is None:
        output_name = f"{args.dataset}_qps_recall_{args.target_recall:.2f}.pdf"
    else:
        output_name = args.output_name

    fig, ax = plt.subplots(figsize=(8, 6))
    legend_handles = []

    for algo_name, algo_folder in ALGORITHMS.items():
        result_files = load_all_results(args.results_dir, args.dataset, algo_folder)
        if not result_files:
            print(f"[SKIP] {algo_name}: no result files found in {args.dataset}/{algo_folder}")
            continue

        best = select_best_config(result_files, args.target_recall)
        if best is None:
            print(f"[SKIP] {algo_name}: could not select a config")
            continue

        entries = sorted(best.entries, key=lambda entry: entry.recall)
        entries = filter_entries_for_plot(algo_name, entries)
        if not entries:
            print(f"[SKIP] {algo_name}: no entries left after applying PLOT_SEARCH_PARAMS")
            continue

        recalls = [entry.recall for entry in entries]
        qps_vals = [entry.qps for entry in entries]
        plotted_params = [entry.search_param for entry in entries]

        style = STYLES.get(algo_name, {"color": "gray", "marker": ".", "linestyle": "-"})
        line, = ax.plot(
            recalls,
            qps_vals,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            label=algo_name,
        )
        legend_handles.append(line)
        print(
            f"[OK]   {algo_name}: selected {os.path.basename(best.filepath)}; "
            f"plotted L values = {plotted_params}"
        )

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    figure_path = os.path.join(dataset_figures_dir, output_name)
    fig.tight_layout()
    fig.savefig(figure_path, format="pdf", bbox_inches="tight")
    print(f"\nFigure saved to {figure_path}")

    if legend_handles:
        fig_leg = plt.figure()
        legend = fig_leg.legend(handles=legend_handles, loc="center", ncol=min(len(legend_handles), 6), frameon=False, fontsize=12)
        fig_leg.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
        legend_path = os.path.join(legends_dir, output_name)
        fig_leg.savefig(legend_path, bbox_inches=bbox, pad_inches=0)
        print(f"Legend saved to {legend_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
