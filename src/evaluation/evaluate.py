"""Compare the baseline and ResNet-18 on the held-out test split, then draw Grad-CAM."""

from __future__ import annotations

import csv
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import ROOT, ensure_dirs, load_manifest
from src.evaluation.gradcam import GradCAM
from src.evaluation.metrics import (
    binary_report,
    bootstrap_auc,
    bootstrap_auc_difference,
    plot_confusion,
    plot_roc,
    youden_threshold,
)
from src.models.cnn import XrayDataset, build_model
from src.preprocessing.ops import read_gray


def _read_predictions(path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y_true = np.asarray([1 if row["label"] == "PNEUMONIA" else 0 for row in rows])
    y_prob = np.asarray([float(row["probability"]) for row in rows])
    return y_true, y_prob, rows


def _select_examples(rows: list[dict], y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> list[dict]:
    y_pred = (y_prob >= threshold).astype(int)
    chosen = []
    for actual, predicted, name in (
        (1, 1, "TP"),
        (0, 0, "TN"),
        (0, 1, "FP"),
        (1, 0, "FN"),
    ):
        indexes = np.flatnonzero((y_true == actual) & (y_pred == predicted))
        if len(indexes) == 0:
            continue
        # Prefer the most confident mistakes and the most confident correct calls.
        order = np.argsort(y_prob[indexes])
        pick = indexes[order[-1]] if predicted == 1 else indexes[order[0]]
        chosen.append({"index": int(pick), "cell": name, "image_id": rows[pick]["image_id"]})
    return chosen


def _gradcam_figure(examples: list[dict], rows: list[dict], model: torch.nn.Module, path) -> None:
    dataset = XrayDataset(rows, train=False)
    cam_engine = GradCAM(model)
    model.eval()
    figure, axes = plt.subplots(2, len(examples), figsize=(3.1 * len(examples), 6.4))
    if len(examples) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for column, example in enumerate(examples):
        image_tensor, _label = dataset[example["index"]]
        predicted_class = 1 if example["cell"] in {"TP", "FP"} else 0
        heatmap = cam_engine(image_tensor, predicted_class)
        gray = read_gray(ROOT / rows[example["index"]]["processed_path"])
        color = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(base, 0.62, color, 0.38, 0)
        axes[0, column].imshow(gray, cmap="gray")
        axes[0, column].set_title(f"{example['cell']} {example['image_id']}", fontsize=8)
        axes[1, column].imshow(overlay)
        axes[1, column].set_title("Grad-CAM", fontsize=8)
        for axis in (axes[0, column], axes[1, column]):
            axis.axis("off")
    figure.suptitle("ResNet-18 Grad-CAM on held-out test images")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate() -> dict:
    paths = ensure_dirs()
    manifest = {row["image_id"]: row for row in load_manifest(paths["reports"] / "manifest.csv")}
    reports = {}
    roc_curves = []
    heldout = {}
    for name, stem in (("logistic_regression", "baseline"), ("resnet18", "cnn")):
        y_val, p_val, _val_rows = _read_predictions(paths["reports"] / f"predictions_{stem}_val.csv")
        y_test, p_test, test_rows = _read_predictions(paths["reports"] / f"predictions_{stem}_test.csv")
        threshold = youden_threshold(y_val, p_val)
        report = binary_report(y_test, p_test, threshold)
        report["model"] = name
        report["at_threshold_0_5"] = binary_report(y_test, p_test, 0.5)
        report["auc_bootstrap95"] = bootstrap_auc(y_test, p_test)
        reports[name] = report
        heldout[name] = (y_test, p_test, [row["image_id"] for row in test_rows])
        roc_curves.append((name, y_test, p_test))
        plot_confusion(
            np.asarray(report["confusion_matrix"]),
            paths["figures"] / f"confusion_{stem}.png",
            f"{name} test, threshold {threshold:.3f}",
        )
        if name == "resnet18":
            ordered_rows = [manifest[row["image_id"]] for row in test_rows]
            examples = _select_examples(test_rows, y_test, p_test, threshold)
            model = build_model()
            state_dict = torch.load(paths["models"] / "resnet18_best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            _gradcam_figure(examples, ordered_rows, model, paths["figures"] / "gradcam.png")
            report["gradcam_examples"] = [{key: value for key, value in item.items() if key != "index"} for item in examples]

    y_test, p_cnn, ids_cnn = heldout["resnet18"]
    y_base, p_base, ids_base = heldout["logistic_regression"]
    if ids_cnn != ids_base or not np.array_equal(y_test, y_base):
        raise RuntimeError("test predictions are not aligned across models")
    reports["auc_difference_resnet_minus_logistic"] = bootstrap_auc_difference(y_test, p_cnn, p_base)
    plot_roc(roc_curves, paths["figures"] / "roc_test.png")
    comparison_path = paths["reports"] / "comparison.csv"
    fieldnames = [
        "model",
        "auc_roc",
        "sensitivity",
        "specificity",
        "accuracy",
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for report in reports.values():
            writer.writerow(report)
    (paths["reports"] / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return reports


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
