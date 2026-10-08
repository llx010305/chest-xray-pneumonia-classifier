"""Resize, denoise with the C++ tool, and normalize the working set."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from src.common import ROOT, ensure_dirs, load_config, load_manifest, save_manifest
from src.preprocessing.ops import median3, normalize_intensity, read_gray, read_pgm, resize_square, write_image, write_pgm

CPP_SOURCE = ROOT / "cpp" / "median_denoise.cpp"
CPP_BINARY = ROOT / "cpp" / "median_denoise.exe"


def compile_denoiser() -> Path:
    if CPP_BINARY.exists() and CPP_BINARY.stat().st_mtime >= CPP_SOURCE.stat().st_mtime:
        return CPP_BINARY
    command = ["g++", "-O2", "-std=c++17", "-o", str(CPP_BINARY), str(CPP_SOURCE)]
    print("compiling", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return CPP_BINARY


def run_denoiser(binary: Path, input_dir: Path, output_dir: Path) -> float:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    subprocess.run([str(binary), str(input_dir), str(output_dir)], check=True)
    return time.perf_counter() - started


def benchmark(binary: Path, images: list[np.ndarray], staging: Path) -> dict:
    bench_in = staging / "bench_in"
    bench_out = staging / "bench_out"
    if bench_in.exists():
        shutil.rmtree(bench_in)
    bench_in.mkdir(parents=True)
    for index, image in enumerate(images):
        write_pgm(bench_in / f"{index:04d}.pgm", image)

    cpp_seconds = run_denoiser(binary, bench_in, bench_out)
    started = time.perf_counter()
    numpy_outputs = [median3(image) for image in images]
    numpy_seconds = time.perf_counter() - started

    max_abs_diff = 0
    for index, expected in enumerate(numpy_outputs):
        actual = read_pgm(bench_out / f"{index:04d}.pgm")
        max_abs_diff = max(max_abs_diff, int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))))
    compiler = subprocess.run(["g++", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return {
        "images": len(images),
        "height": int(images[0].shape[0]),
        "width": int(images[0].shape[1]),
        "kernel": "3x3 median, replicate border",
        "cpp_seconds": cpp_seconds,
        "numpy_seconds": numpy_seconds,
        "cpp_images_per_second": len(images) / cpp_seconds if cpp_seconds else None,
        "numpy_images_per_second": len(images) / numpy_seconds if numpy_seconds else None,
        "max_abs_diff": max_abs_diff,
        "compiler": compiler,
    }


def preprocess(config: dict | None = None) -> dict:
    config = config or load_config()
    paths = ensure_dirs()
    manifest_path = paths["reports"] / "manifest.csv"
    rows = load_manifest(manifest_path)
    size = int(config["image_size"])
    staging = paths["interim"] / "pgm_resized"
    denoised_dir = paths["interim"] / "pgm_denoised"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    resized: list[np.ndarray] = []
    for row in rows:
        image = resize_square(read_gray(ROOT / row["selected_path"]), size)
        resized.append(image)
        write_pgm(staging / f"{row['image_id']}.pgm", image)

    binary = compile_denoiser()
    benchmark_n = min(int(config["benchmark_images"]), len(resized))
    benchmark_report = benchmark(binary, resized[:benchmark_n], paths["interim"])
    if benchmark_report["max_abs_diff"] != 0:
        raise RuntimeError(f"C++ median does not match NumPy: {benchmark_report}")

    production_seconds = run_denoiser(binary, staging, denoised_dir)
    for row in rows:
        denoised = read_pgm(denoised_dir / f"{row['image_id']}.pgm")
        normalized = normalize_intensity(
            denoised,
            low_percentile=float(config["percentile_low"]),
            high_percentile=float(config["percentile_high"]),
        )
        relative = Path("data") / "processed" / "images" / f"{row['image_id']}.png"
        write_image(ROOT / relative, normalized)
        row["processed_path"] = relative.as_posix()

    save_manifest(manifest_path, rows)
    summary = {
        "images": len(rows),
        "image_size": size,
        "percentile_low": float(config["percentile_low"]),
        "percentile_high": float(config["percentile_high"]),
        "production_cpp_seconds": production_seconds,
        "benchmark": benchmark_report,
    }
    summary_path = paths["reports"] / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    preprocess()


if __name__ == "__main__":
    main()
