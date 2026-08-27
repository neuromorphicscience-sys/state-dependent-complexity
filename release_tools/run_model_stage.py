#!/usr/bin/env python3
"""Cross-platform launcher for the self-contained Stage 1-4A model packages."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = RELEASE_ROOT / "analysis" / "model_stages"


@dataclass(frozen=True)
class Stage:
    directory: str
    runner: str
    config: str
    smoke_config: str | None = None
    analyzer: str | None = None
    phase_argument: bool = False


STAGES = {
    "stage1": Stage(
        "stage01", "scripts/run_sweep.py", "configs/stage1_baseline.json",
        "configs/smoke.json", "scripts/summarize.py",
    ),
    "stage1b": Stage(
        "stage01b", "scripts/run_stage1b.py", "configs/stage1b.json",
        analyzer="scripts/analyze_stage1b.py",
    ),
    "stage2": Stage(
        "stage02", "scripts/run_stage2.py", "configs/stage2.json",
        "configs/stage2_smoke.json", "scripts/analyze_stage2.py",
    ),
    "stage3": Stage(
        "stage03", "scripts/run_stage3.py", "configs/stage3.json",
        "configs/stage3_smoke.json", "scripts/analyze_stage3.py",
    ),
    "stage4": Stage(
        "stage04", "scripts/run_stage4.py", "configs/stage4.json",
        "configs/stage4_smoke.json", "scripts/analyze_stage4.py", True,
    ),
    "stage4a": Stage(
        "stage04a/formal_frozen", "scripts/run_stage4.py",
        "configs/stage4a.json", phase_argument=True,
    ),
}


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"cwd: {cwd}")
    print(f"run: {command_text(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter to use (default: the current interpreter).",
    )
    parser.add_argument("--config", type=Path, help="Override the stage config.")
    parser.add_argument("--smoke", action="store_true", help="Use the stage smoke config.")
    parser.add_argument("--phase", choices=("discovery", "heldout"), default="discovery")
    parser.add_argument("--analyze", action="store_true", help="Run the stage analyzer after compute.")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = STAGES[args.stage]
    stage_dir = MODEL_ROOT / spec.directory
    if args.stage == "stage4a" and args.phase != "discovery":
        raise SystemExit("Stage 4A's frozen formal workflow is discovery-only.")
    if args.smoke and not spec.smoke_config:
        raise SystemExit(f"No smoke config is archived for {args.stage}.")

    config = args.config or Path(spec.smoke_config if args.smoke else spec.config)
    if not config.is_absolute():
        config = stage_dir / config
    runner = stage_dir / spec.runner
    gpu_check = stage_dir / "scripts" / "check_gpu.py"
    for required in (stage_dir, runner, config):
        if not required.exists():
            raise SystemExit(f"Required path does not exist: {required}")

    if args.stage == "stage4":
        print("warning: stage4 uses the legacy/general config; use stage4a for frozen Stage 4A.")

    if not args.skip_gpu_check and gpu_check.exists():
        run([args.python, str(gpu_check)], stage_dir, args.dry_run)
    command = [args.python, str(runner), "--config", str(config)]
    if spec.phase_argument:
        command += ["--phase", args.phase]
    run(command, stage_dir, args.dry_run)

    if args.analyze:
        if not spec.analyzer:
            raise SystemExit(
                f"No stage-local analyzer is archived for {args.stage}; "
                "see docs/STAGE_CODE_MAP.md for the final analysis entry point."
            )
        analyzer = stage_dir / spec.analyzer
        run([args.python, str(analyzer)], stage_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

