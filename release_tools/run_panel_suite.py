#!/usr/bin/env python3
"""Run the V5.3 panel suite with an explicit, portable external data root."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
BUILDER = RELEASE_ROOT / "plotting" / "panel_suite_v5_3" / "build_panel_suite_v5_3.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=os.environ.get("NEURAL_SCIENCE_DATA_ROOT"),
        help="External project/data root, or set NEURAL_SCIENCE_DATA_ROOT.",
    )
    args, remainder = parser.parse_known_args()
    if args.data_root is None:
        parser.error("provide --data-root or set NEURAL_SCIENCE_DATA_ROOT")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"data root does not exist: {data_root}")
    command = [sys.executable, str(BUILDER), "--root", str(data_root), *remainder]
    print("run:", subprocess.list2cmdline(command))
    return subprocess.run(command, cwd=BUILDER.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())

