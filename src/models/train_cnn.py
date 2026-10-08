"""Fine-tune ImageNet ResNet-18. The checkpoint is chosen by validation AUC."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.common import ensure_dirs, load_config, load_manifest
from src.models.cnn import XrayDataset, build_model, configure_torch_home


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(rows: list[dict], train: bool, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        XrayDataset(rows, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        generator=generator,
    )


def _predict(model: torch.nn.Module, rows: list[dict], batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(XrayDataset(rows, train=False), batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


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


def train_cnn(config: dict | None = None) -> dict:
    config = config or load_config()
    configure_torch_home()
    set_seed(int(config["seed"]))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    paths = ensure_dirs()
    rows = load_manifest(paths["reports"] / "manifest.csv")
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    training = config["training"]
    batch_size = int(training["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training ResNet-18 on {device}")
    model = build_model().to(device)
    backbone = [param for name, param in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(training["lr_backbone"])},
            {"params": model.fc.parameters(), "lr": float(training["lr_head"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    # The working set is sampled to be nearly balanced; balanced class weights stay correct if a split drifts.
    train_labels = [1 if row["label"] == "PNEUMONIA" else 0 for row in splits["train"]]
    positive = max(sum(train_labels), 1)
    negative = max(len(train_labels) - positive, 1)
    weights = torch.tensor([len(train_labels) / (2 * negative), len(train_labels) / (2 * positive)], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    train_loader = _loader(splits["train"], train=True, batch_size=batch_size, seed=int(config["seed"]))
    y_val = np.asarray([1 if row["label"] == "PNEUMONIA" else 0 for row in splits["val"]])

    best_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint = paths["models"] / "resnet18_best.pt"
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_prob = _predict(model, splits["val"], batch_size, device)
        # _predict sets eval mode. The next epoch calls train() again.
        val_auc = float(roc_auc_score(y_val, val_prob))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_auc": val_auc,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch}: loss {record['train_loss']:.4f} val_auc {val_auc:.4f} ({record['seconds']:.1f}s)",
            flush=True,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            stale_epochs = 0
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print("early stopping")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    _write_predictions(
        paths["reports"] / "predictions_cnn_val.csv",
        splits["val"],
        _predict(model, splits["val"], batch_size, device),
    )
    _write_predictions(
        paths["reports"] / "predictions_cnn_test.csv",
        splits["test"],
        _predict(model, splits["test"], batch_size, device),
    )
    summary = {"best_val_auc": best_auc, "best_epoch": best_epoch, "device": str(device), "history": history}
    (paths["reports"] / "cnn_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("best_val_auc", "best_epoch", "device")}, indent=2))
    return summary


def main() -> None:
    train_cnn()


if __name__ == "__main__":
    main()
