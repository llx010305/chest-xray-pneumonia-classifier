"""PGM round trip, the NumPy median reference, normalization, and the C++ filter."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common import ROOT
from src.preprocessing.ops import median3, normalize_intensity, read_pgm, resize_square, write_pgm


def _slow_median3(image: np.ndarray) -> np.ndarray:
    """Loop-based 3x3 median with edge replication, written independently of median3."""
    height, width = image.shape
    output = np.empty_like(image)
    for y in range(height):
        for x in range(width):
            window = [
                image[min(max(y + dy, 0), height - 1), min(max(x + dx, 0), width - 1)]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            output[y, x] = sorted(window)[4]
    return output


class PgmTest(unittest.TestCase):
    def test_round_trip_keeps_every_byte(self):
        rng = np.random.default_rng(0)
        # Include byte values that look like PGM whitespace (9, 10, 13, 32).
        image = rng.integers(0, 256, size=(17, 23), dtype=np.uint8)
        image[0, :4] = [9, 10, 13, 32]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.pgm"
            write_pgm(path, image)
            np.testing.assert_array_equal(read_pgm(path), image)

    def test_writer_rejects_non_uint8(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_pgm(Path(folder) / "bad.pgm", np.zeros((4, 4), dtype=np.float32))


class MedianTest(unittest.TestCase):
    def test_matches_independent_loop(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(12, 15), dtype=np.uint8)
        np.testing.assert_array_equal(median3(image), _slow_median3(image))

    def test_removes_isolated_salt_noise(self):
        image = np.full((9, 9), 100, dtype=np.uint8)
        image[4, 4] = 255
        self.assertEqual(int(median3(image)[4, 4]), 100)

    def test_keeps_shape_and_dtype(self):
        image = np.zeros((224, 224), dtype=np.uint8)
        result = median3(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)


class NormalizeTest(unittest.TestCase):
    def test_stretches_to_full_range(self):
        image = np.tile(np.arange(50, 151, dtype=np.uint8), (10, 1))
        result = normalize_intensity(image, 1.0, 99.0)
        self.assertEqual(int(result.min()), 0)
        self.assertEqual(int(result.max()), 255)
        self.assertEqual(result.dtype, np.uint8)

    def test_flat_image_does_not_divide_by_zero(self):
        result = normalize_intensity(np.full((8, 8), 77, dtype=np.uint8))
        self.assertTrue(np.all(np.isfinite(result.astype(float))))

    def test_resize_square(self):
        self.assertEqual(resize_square(np.zeros((300, 500), dtype=np.uint8), 224).shape, (224, 224))
        self.assertEqual(resize_square(np.zeros((100, 120), dtype=np.uint8), 224).shape, (224, 224))


@unittest.skipUnless(shutil.which("g++"), "g++ is not installed")
class CppMedianTest(unittest.TestCase):
    """Compile cpp/median_denoise.cpp into a temp folder and compare with median3."""

    def test_cpp_matches_numpy(self):
        rng = np.random.default_rng(2)
        images = [rng.integers(0, 256, size=(31, 29), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            binary = folder / "median_denoise"
            subprocess.run(
                ["g++", "-O2", "-std=c++17", "-o", str(binary), str(ROOT / "cpp" / "median_denoise.cpp")],
                check=True,
            )
            (folder / "in").mkdir()
            for index, image in enumerate(images):
                write_pgm(folder / "in" / f"{index}.pgm", image)
            subprocess.run([str(binary), str(folder / "in"), str(folder / "out")], check=True, capture_output=True)
            for index, image in enumerate(images):
                np.testing.assert_array_equal(read_pgm(folder / "out" / f"{index}.pgm"), median3(image))


if __name__ == "__main__":
    unittest.main()
