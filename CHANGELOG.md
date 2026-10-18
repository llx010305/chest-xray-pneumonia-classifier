# Changelog

All notable changes to this project. Dates are when the change was published to this repository.

## [Unreleased]

### Added
- Calibration check: Brier score and a 10-bin reliability diagram (`reports/figures/calibration_test.png`) for both models on the test set, stored under `calibration` in `reports/metrics.json`.
- 95% Wilson intervals for test sensitivity and specificity at both thresholds (`sensitivity_wilson95`, `specificity_wilson95` in `reports/metrics.json`).

## [2.0] – 2026-10-11

An audit of the v1.0 working set found near-duplicate films that the exact 16×16 match had missed. This release rebuilds the data and every result from scratch. **v1.0 numbers should not be cited.**

### Found
- 1,576 clusters of copied films across the two archives (8,482 images). In every pair checked, copies differed only by re-encoding (one grey level in a few thumbnail pixels) or resizing.
- All 1,306 Chittagong v4 pneumonia images are copies of Kermany films; 337 of them copy films labelled **normal** in Kermany.
- In the v1.0 working set: 292 surviving duplicate pairs, 117 with opposite labels, and 85 test images with a copy in train or validation.

### Changed
- De-duplication links files by SHA-256, identical 16×16 thumbnail, or 32×32 thumbnail correlation ≥ 0.995 (`dedup_correlation`), clusters them, keeps one copy, and drops every copy when labels conflict. Decisions are written to `reports/duplicate_clusters.csv`.
- Train/validation/test now use Kermany only (3,000 images: 1,242 normal, 1,758 pneumonia). Mixing in Chittagong, which only contributes normals after de-duplication, would let the source act as a label shortcut.
- The 1,006 unique Chittagong normal films form an external set. Both models score it, and `evaluate.py` reports specificity with a Wilson interval and draws `external_normals.png`.
- EDA describes the working set only.
- `requirements.txt` pins the versions used for the recorded run (Python 3.10, PyTorch 2.10 CPU, OpenCV 4.10).

### Fixed
- Empty trailing row in `reports/comparison.csv`.
- C++ binary is named per platform (`median_denoise` on macOS/Linux, `.exe` on Windows) and is rebuilt if it will not run.
- Paths in the manifest and summaries use `/` on every platform.
- Confusion-matrix counts are readable on dark cells.
- Evaluation no longer downloads ImageNet weights that the fine-tuned checkpoint replaces.

### Added
- `tests/` (39 unit tests) and `scripts/run_pipeline.py`.

### Results
| | v1.0 (superseded) | v2.0 |
| --- | ---: | ---: |
| Logistic regression, test AUC | 0.908 | 0.986 |
| ResNet-18, test AUC | 0.924 | 0.989 |
| ResNet-18 specificity on external normals | – | 0.261 |

## [1.0] – 2026-10-08

First public version: two Mendeley archives merged with exact-match de-duplication, 3,000 images (750 per source and class), logistic regression vs. ResNet-18, C++ median filter, bootstrap CIs, and Grad-CAM.
