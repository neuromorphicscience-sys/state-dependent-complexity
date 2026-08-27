#!/usr/bin/env python3
"""Portable launcher for the Stage 5C/SynPhys analysis phases."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = RELEASE_ROOT / "analysis/biological_stages/stage05c_synphys/phase_scripts"
ORDER = ("schema", "phase1", "transfer", "model_audit", "pairlevel", "dynamic", "identity")


def phase_command(phase: str, synphys: Path, python: str) -> list[str]:
    db = synphys / "synphys_r2.1_full.sqlite"
    phase1 = synphys / "stage5c_synphys_phase1"
    transfer = synphys / "stage5c_synphys_transfer_v2"
    audit = synphys / "stage5c_synapse_model_audit_v1"
    dynamic = synphys / "stage5c_dynamic_leverage_v4"
    commands = {
        "schema": [python, str(SCRIPTS / "stage5c_synphys_schema_audit.py"), "--db", str(db)],
        "phase1": [
            python, str(SCRIPTS / "stage5c_synphys_phase1.py"),
            "--db", str(db), "--out", str(phase1),
        ],
        "transfer": [
            python, str(SCRIPTS / "stage5c_frozen_transfer_v2.py"),
            "--root", str(synphys.parent), "--phase1", str(phase1),
            "--out", str(transfer),
        ],
        "model_audit": [
            python, str(SCRIPTS / "stage5c_synapse_model_audit_v1.py"),
            "--db", str(db),
            "--transfer", str(transfer / "synphys_frozen_complexity_transfer.csv"),
            "--out", str(audit),
        ],
        "pairlevel": [
            python, str(SCRIPTS / "stage5c_pairlevel_bridge_v3.py"),
            "--phase1", str(phase1), "--transfer", str(transfer),
            "--out", str(synphys / "stage5c_synphys_pairlevel_v3"),
        ],
        "dynamic": [
            python, str(SCRIPTS / "stage5c_dynamic_leverage_v4.py"),
            "--audit", str(audit), "--phase1", str(phase1),
            "--out", str(dynamic),
        ],
        "identity": [
            python, str(SCRIPTS / "stage5c_identity_decomposition_v5.py"),
            "--v4", str(dynamic),
            "--out", str(synphys / "stage5c_identity_decomposition_v5"),
        ],
    }
    return commands[phase]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*ORDER, "all"))
    parser.add_argument(
        "--synphys-root", type=Path, required=True,
        help="Directory containing synphys_r2.1_full.sqlite and Stage 5C outputs.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    synphys = args.synphys_root.expanduser().resolve()
    phases = ORDER if args.phase == "all" else (args.phase,)
    for phase in phases:
        command = phase_command(phase, synphys, args.python)
        print(f"[{phase}] {shlex.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=SCRIPTS, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

