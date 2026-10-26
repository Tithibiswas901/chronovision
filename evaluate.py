"""
evaluate.py
-----------
Batch-evaluates all three unsupervised change-detection methods
(image_difference_otsu, change_vector_analysis, pca_kmeans) over every
bi-temporal image pair in `data/`, against the LEVIR-CD ground-truth
labels.

Outputs:
    results.csv            Per-(pair, method) metric values.
    results_summary.csv    Mean of each metric, grouped by method.
    figures/metrics_bar.png   Grouped bar chart comparing F1 / IoU / Kappa.
    figures/qualitative.png   Visual comparison for 4 sample pairs:
                              A | B | Ground Truth | Diff+Otsu | CVA | PCA-KMeans

Run with:
    python evaluate.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend: safe for running without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import compute_metrics
from src.methods import change_vector_analysis, image_difference_otsu, pca_kmeans
from src.postprocessing import clean_change_map
from src.preprocessing import load_and_preprocess_pair

DATA_DIR = "data"
DIR_A = os.path.join(DATA_DIR, "A")
DIR_B = os.path.join(DATA_DIR, "B")
DIR_LABEL = os.path.join(DATA_DIR, "label")
FIGURES_DIR = "figures"

# Methods to evaluate: (display name, callable).
METHODS = [
    ("Diff+Otsu", image_difference_otsu),
    ("CVA", change_vector_analysis),
    ("PCA-KMeans", pca_kmeans),
]

N_QUALITATIVE_SAMPLES = 4


def list_pairs():
    """Return the sorted list of filenames present in all of A/, B/, label/."""
    names_a = set(os.listdir(DIR_A))
    names_b = set(os.listdir(DIR_B))
    names_label = set(os.listdir(DIR_LABEL))
    common = sorted(names_a & names_b & names_label)
    if not common:
        raise RuntimeError(f"No matching filenames found across {DIR_A}, {DIR_B}, {DIR_LABEL}")
    return common


def run_all_methods(img_a, img_b):
    """Run every method in METHODS on a preprocessed pair, with postprocessing.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of method display name -> cleaned binary change map.
    """
    outputs = {}
    for name, fn in METHODS:
        raw_map = fn(img_a, img_b)
        outputs[name] = clean_change_map(raw_map)
    return outputs


def evaluate_all_pairs(filenames):
    """Run preprocessing + all methods + metrics for every pair.

    Returns
    -------
    pd.DataFrame
        One row per (filename, method) with all metric columns.
    dict[str, dict]
        Mapping filename -> preprocessed data + per-method change maps,
        kept in memory for the qualitative figure (only needed for the
        first N_QUALITATIVE_SAMPLES filenames to limit memory use).
    """
    rows = []
    qualitative_cache = {}

    for i, fname in enumerate(filenames):
        path_a = os.path.join(DIR_A, fname)
        path_b = os.path.join(DIR_B, fname)
        path_label = os.path.join(DIR_LABEL, fname)

        data = load_and_preprocess_pair(path_a, path_b, path_label)
        img_a, img_b, label = data["a"], data["b"], data["label"]

        change_maps = run_all_methods(img_a, img_b)

        for method_name, change_map in change_maps.items():
            metrics = compute_metrics(change_map, label)
            rows.append({"filename": fname, "method": method_name, **metrics})

        print(f"[{i + 1}/{len(filenames)}] evaluated {fname}")

        if i < N_QUALITATIVE_SAMPLES:
            qualitative_cache[fname] = {
                "a": img_a,
                "b": img_b,
                "label": label,
                "maps": change_maps,
            }

    return pd.DataFrame(rows), qualitative_cache


def save_metrics_bar_chart(summary_df, out_path):
    """Save a grouped bar chart comparing methods on F1, IoU, and Kappa."""
    metrics_to_plot = ["f1", "iou", "kappa"]
    methods = summary_df["method"].tolist()

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        values = summary_df.loc[summary_df["method"] == method, metrics_to_plot].values.flatten()
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=method)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(["F1", "IoU", "Kappa"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Change Detection Method Comparison (mean over dataset)")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_qualitative_figure(qualitative_cache, out_path):
    """Save a grid figure: rows = sample pairs, columns = A | B | GT | each method."""
    filenames = list(qualitative_cache.keys())[:N_QUALITATIVE_SAMPLES]
    n_rows = len(filenames)
    col_titles = ["Image 1 (A)", "Image 2 (B)", "Ground Truth"] + [name for name, _ in METHODS]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, fname in enumerate(filenames):
        sample = qualitative_cache[fname]
        images = [sample["a"], sample["b"], sample["label"]] + [
            sample["maps"][name] for name, _ in METHODS
        ]

        for col, (img, title) in enumerate(zip(images, col_titles)):
            ax = axes[row, col]
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(fname, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Qualitative Comparison of Change Detection Methods", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    filenames = list_pairs()
    print(f"Found {len(filenames)} image pairs in '{DATA_DIR}'.")

    results_df, qualitative_cache = evaluate_all_pairs(filenames)
    results_df.to_csv("results.csv", index=False)
    print("\nSaved per-pair, per-method metrics to results.csv")

    summary_df = (
        results_df.groupby("method")[["precision", "recall", "f1", "iou", "accuracy", "kappa"]]
        .mean()
        .reset_index()
    )
    summary_df.to_csv("results_summary.csv", index=False)
    print("Saved method-averaged metrics to results_summary.csv")

    save_metrics_bar_chart(summary_df, os.path.join(FIGURES_DIR, "metrics_bar.png"))
    print(f"Saved bar chart to {FIGURES_DIR}/metrics_bar.png")

    save_qualitative_figure(qualitative_cache, os.path.join(FIGURES_DIR, "qualitative.png"))
    print(f"Saved qualitative figure to {FIGURES_DIR}/qualitative.png")

    print("\n=== Summary (mean metrics per method) ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
