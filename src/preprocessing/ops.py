"""NumPy reference operations used by preprocessing and the C++ benchmark."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    """Read a grayscale image. np.fromfile keeps non-ASCII paths working on Windows."""
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if min(image.shape[:2]) >= size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def median3(image: np.ndarray) -> np.ndarray:
    """3x3 median with edge replication. Matches cpp/median_denoise.cpp."""
    if image.ndim != 2:
        raise ValueError("median3 expects a grayscale image")
    padded = np.pad(image, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1)).astype(np.uint8)


def normalize_intensity(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Clip to a percentile window and scale that window onto 0-255."""
    low = float(np.percentile(image, low_percentile))
    high = float(np.percentile(image, high_percentile))
    if high <= low:
        high = low + 1.0
    scaled = (image.astype(np.float32) - low) / (high - low)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def write_pgm(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("PGM writer expects a uint8 grayscale image")
    height, width = image.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + image.tobytes())


def read_pgm(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if not payload.startswith(b"P5"):
        raise ValueError(f"not a P5 PGM: {path}")
    rest = payload[2:]
    fields = []
    cursor = 0
    while len(fields) < 3:
        while cursor < len(rest) and rest[cursor] in b" \t\r\n":
            cursor += 1
        start = cursor
        while cursor < len(rest) and rest[cursor] not in b" \t\r\n":
            cursor += 1
        fields.append(rest[start:cursor])
    width = int(fields[0])
    height = int(fields[1])
    if int(fields[2]) != 255:
        raise ValueError("only 8-bit PGM is supported")
    # P5 allows exactly one whitespace separator. Later bytes are pixels,
    # including values that look like spaces or newlines.
    if cursor < len(rest) and rest[cursor] in b" \t\r\n":
        cursor += 1
    pixels = np.frombuffer(rest[cursor:], dtype=np.uint8)
    if pixels.size != width * height:
        raise ValueError(f"unexpected PGM size: {path}")
    return pixels.reshape(height, width).copy()
