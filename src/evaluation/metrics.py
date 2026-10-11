"""Threshold-free and operating-point metrics for pneumonia as the positive class."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.eda.explore import wilson_interval


def bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> dict:
    """Percentile bootstrap confidence interval for ROC AUC."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[index]).size < 2:
            continue
        scores.append(roc_auc_score(y_true[index], y_prob[index]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": len(scores),
    }


def bootstrap_auc_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap interval for AUC(a) - AUC(b) on the same test images."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[index]).size < 2:
            continue
        scores.append(roc_auc_score(y_true[index], y_prob_a[index]) - roc_auc_score(y_true[index], y_prob_b[index]))
    low, high = np.percentile(scores, [2.5, 97.5])
    point = float(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b))
    return {"difference": point, "ci95_low": float(low), "ci95_high": float(high), "resamples": len(scores)}


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose the threshold that maximizes sensitivity + specificity - 1 on validation data."""
    _fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve may return a threshold above 1 for the first point. Ignore non-finite values.
    finite = np.isfinite(thresholds)
    scores = tpr[finite] - _fpr[finite]
    return float(thresholds[finite][np.argmax(scores)])


def binary_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    accuracy = (true_positive + true_negative) / matrix.sum()
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": int(true_positive),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "confusion_matrix": matrix.tolist(),
    }


def negatives_report(y_prob: np.ndarray, threshold: float) -> dict:
    """Specificity on a set that contains only normal films (no positives to score).

    Every call at or above the threshold is a false positive. The 95% Wilson
    interval is for the specificity.
    """
    total = int(len(y_prob))
    false_positive = int((y_prob >= threshold).sum())
    true_negative = total - false_positive
    low, high = wilson_interval(true_negative, total)
    return {
        "threshold": float(threshold),
        "images": total,
        "fp": false_positive,
        "tn": true_negative,
        "specificity": true_negative / total,
        "specificity_wilson95": [low, high],
    }


def plot_normal_scores(panels: list[tuple[str, np.ndarray, np.ndarray, float]], path) -> None:
    """Pneumonia probability on normal films: held-out Kermany test vs. the external set.

    Each panel is (model name, Kermany test normal probabilities, external
    probabilities, operating threshold).
    """
    figure, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.0), squeeze=False)
    bins = np.linspace(0, 1, 26)
    for axis, (name, internal, external, threshold) in zip(axes[0], panels):
        axis.hist(internal, bins=bins, density=True, alpha=0.6, label=f"Kermany test normals (n={len(internal)})")
        axis.hist(external, bins=bins, density=True, alpha=0.6, label=f"Chittagong normals (n={len(external)})")
        axis.axvline(threshold, color="#444444", linestyle="--", linewidth=1, label=f"threshold {threshold:.3f}")
        axis.set_xlabel("Predicted probability of pneumonia")
        axis.set_ylabel("Density")
        axis.set_title(name)
        axis.legend(fontsize=8)
    figure.suptitle("Normal films only: scores right of the line are false positives")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion(matrix: np.ndarray, path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Normal", "Pneumonia"])
    axis.set_yticks([0, 1], ["Normal", "Pneumonia"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            # White text on dark cells, black on light ones.
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=color, fontsize=12)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray]], path) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    for name, y_true, y_prob in curves:
        fpr, tpr, _thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axis.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_title("Test ROC")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
