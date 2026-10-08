"""Shared paths and configuration."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    return config


def load_manifest(path: Path | None = None) -> list[dict]:
    manifest = path or (ROOT / "reports" / "manifest.csv")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot save an empty manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "raw": ROOT / "data" / "raw",
        "interim": ROOT / "data" / "interim",
        "processed": ROOT / "data" / "processed" / "images",
        "reports": ROOT / "reports",
        "figures": ROOT / "reports" / "figures",
        "models": ROOT / "reports" / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
