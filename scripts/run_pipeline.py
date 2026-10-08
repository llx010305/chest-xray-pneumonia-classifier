"""Run the whole pipeline in order from the repository root.

Usage:
    python scripts/run_pipeline.py                 # every step
    python scripts/run_pipeline.py --from baseline # resume from a step
    python scripts/run_pipeline.py --skip-download # archives already in data/raw
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("download", "src.data.download"),
    ("prepare", "src.data.prepare"),
    ("preprocess", "src.preprocessing.preprocess"),
    ("eda", "src.eda.explore"),
    ("baseline", "src.models.baseline"),
    ("cnn", "src.models.train_cnn"),
    ("evaluate", "src.evaluation.evaluate"),
]


def main() -> int:
    names = [name for name, _module in STEPS]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=names, default=names[0], help="first step to run")
    parser.add_argument("--skip-download", action="store_true", help="skip the download step")
    args = parser.parse_args()

    steps = STEPS[names.index(args.start):]
    if args.skip_download:
        steps = [step for step in steps if step[0] != "download"]

    for name, module in steps:
        print(f"\n=== {name}: python -m {module}", flush=True)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"step '{name}' failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
        print(f"=== {name} finished in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
