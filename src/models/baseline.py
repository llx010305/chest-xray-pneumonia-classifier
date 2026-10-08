"""Logistic regression on HOG, intensity histogram, and downsampled pixels."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, ensure_dirs, load_config, load_manifest
from src.preprocessing.ops import read_gray

HOG = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)


def handcrafted_features(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hog = HOG.compute(small).ravel()
    histogram, _edges = np.histogram(image, bins=32, range=(0, 256), density=True)
    mean = float(image.mean()) / 255.0
    std = float(image.std()) / 255.0
    centered = image.astype(np.float32) - image.mean()
    skew = float((centered**3).mean() / (image.std() ** 3 + 1e-6))
    pooled = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA).ravel().astype(np.float32) / 255.0
    return np.concatenate([hog, histogram.astype(np.float32), np.array([mean, std, skew], dtype=np.float32), pooled])


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for index, row in enumerate(rows):
        image = read_gray(ROOT / row["processed_path"])
        features.append(handcrafted_features(image))
        labels.append(1 if row["label"] == "PNEUMONIA" else 0)
        if index and index % 400 == 0:
            print(f"  features {index}/{len(rows)}", flush=True)
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def _write_predictions(path: Path, rows: list[dict], probabilities: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "label", "probability"])
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "probability": f"{float(probability):.8f}",
                }
            )


def train_baseline(config: dict | None = None) -> None:
    config = config or load_config()
    paths = ensure_dirs()
    rows = load_manifest(paths["reports"] / "manifest.csv")
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    print(f"extracting features for {len(rows)} images")
    x_train, y_train = _matrix(splits["train"])
    x_val, _y_val = _matrix(splits["val"])
    x_test, _y_test = _matrix(splits["test"])
    baseline = config["baseline"]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(baseline["C"]),
                    max_iter=int(baseline["max_iter"]),
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=int(config["seed"]),
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    joblib.dump(model, paths["models"] / "baseline_logreg.joblib")
    _write_predictions(paths["reports"] / "predictions_baseline_val.csv", splits["val"], model.predict_proba(x_val)[:, 1])
    _write_predictions(
        paths["reports"] / "predictions_baseline_test.csv", splits["test"], model.predict_proba(x_test)[:, 1]
    )
    print(f"saved baseline model and predictions; train rows={len(splits['train'])}")


def main() -> None:
    train_baseline()


if __name__ == "__main__":
    main()
