"""Class balance, intensity distributions, and a Welch test between classes."""

from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.common import ROOT, ensure_dirs, load_manifest
from src.preprocessing.ops import read_gray


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("wilson_interval needs at least one trial")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return float(center - margin), float(center + margin)


def welch_difference_ci(left: np.ndarray, right: np.ndarray) -> dict:
    """Welch t-test and 95% CI for mean(left) - mean(right)."""
    test = stats.ttest_ind(left, right, equal_var=False)
    n1, n2 = len(left), len(right)
    mean_left, mean_right = float(left.mean()), float(right.mean())
    var_left, var_right = float(left.var(ddof=1)), float(right.var(ddof=1))
    se_left, se_right = var_left / n1, var_right / n2
    standard_error = float(np.sqrt(se_left + se_right))
    degrees = (se_left + se_right) ** 2 / (se_left**2 / (n1 - 1) + se_right**2 / (n2 - 1))
    critical = float(stats.t.ppf(0.975, degrees))
    difference = mean_left - mean_right
    return {
        "difference_pneumonia_minus_normal": difference,
        "ci95_low": difference - critical * standard_error,
        "ci95_high": difference + critical * standard_error,
        "degrees_of_freedom": float(degrees),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def explore() -> dict:
    paths = ensure_dirs()
    all_rows = load_manifest(paths["reports"] / "manifest.csv")
    if "processed_path" not in all_rows[0]:
        raise RuntimeError("manifest has no processed_path; run preprocessing first")
    # The external set is only for the false-positive check, not for describing the working set.
    rows = [row for row in all_rows if row["split"] != "external"]
    external_rows = len(all_rows) - len(rows)

    means = []
    labels = []
    for row in rows:
        image = read_gray(ROOT / row["processed_path"])
        means.append(float(image.mean()))
        labels.append(row["label"])
    means_array = np.asarray(means)
    labels_array = np.asarray(labels)
    pneumonia = means_array[labels_array == "PNEUMONIA"]
    normal = means_array[labels_array == "NORMAL"]
    mannwhitney = stats.mannwhitneyu(pneumonia, normal, alternative="two-sided")
    pneumonia_count = int((labels_array == "PNEUMONIA").sum())
    normal_count = int((labels_array == "NORMAL").sum())
    low, high = wilson_interval(pneumonia_count, len(rows))
    ratio = max(pneumonia_count, normal_count) / min(pneumonia_count, normal_count)
    expected = np.array([len(rows) / 2, len(rows) / 2])
    chi2, chi_p = stats.chisquare([normal_count, pneumonia_count], f_exp=expected)

    summary = {
        "images": len(rows),
        "external_images_excluded": external_rows,
        "class_counts": {"NORMAL": normal_count, "PNEUMONIA": pneumonia_count},
        "larger_to_smaller_ratio": ratio,
        "within_1_5_ratio": bool(ratio <= 1.5),
        "pneumonia_proportion": pneumonia_count / len(rows),
        "pneumonia_proportion_wilson95": [low, high],
        "chi_square_against_equal_counts": {"statistic": float(chi2), "p_value": float(chi_p)},
        "mean_intensity": {
            "NORMAL": {"mean": float(normal.mean()), "std": float(normal.std(ddof=1))},
            "PNEUMONIA": {"mean": float(pneumonia.mean()), "std": float(pneumonia.std(ddof=1))},
        },
        "welch_t_test_mean_intensity": welch_difference_ci(pneumonia, normal),
        "mannwhitney_u_mean_intensity": {
            "statistic": float(mannwhitney.statistic),
            "p_value": float(mannwhitney.pvalue),
        },
        "by_source_and_label": dict(Counter((row["source"], row["label"]) for row in rows)),
    }
    # JSON keys must be strings.
    summary["by_source_and_label"] = {
        f"{source}|{label}": count for (source, label), count in summary["by_source_and_label"].items()
    }
    (paths["reports"] / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_balance(rows, paths["figures"] / "class_balance.png")
    _plot_intensity(normal, pneumonia, paths["figures"] / "intensity_hist.png")
    print(json.dumps(summary, indent=2))
    return summary


def _plot_balance(rows: list[dict], path) -> None:
    sources = sorted({row["source"] for row in rows})
    normal = [sum(row["source"] == source and row["label"] == "NORMAL" for row in rows) for source in sources]
    pneumonia = [sum(row["source"] == source and row["label"] == "PNEUMONIA" for row in rows) for source in sources]
    positions = np.arange(len(sources))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(positions - width / 2, normal, width, label="Normal", color="#4C78A8")
    ax.bar(positions + width / 2, pneumonia, width, label="Pneumonia", color="#F58518")
    ax.set_xticks(positions, sources)
    ax.set_ylabel("Images")
    ax.set_title(f"Working set: {len(rows)} images by source and class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_intensity(normal: np.ndarray, pneumonia: np.ndarray, path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.linspace(0, 255, 40)
    ax.hist(normal, bins=bins, alpha=0.65, label="Normal", color="#4C78A8", density=True)
    ax.hist(pneumonia, bins=bins, alpha=0.65, label="Pneumonia", color="#F58518", density=True)
    ax.set_xlabel("Mean pixel intensity after preprocessing")
    ax.set_ylabel("Density")
    ax.set_title("Intensity by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    explore()


if __name__ == "__main__":
    main()
