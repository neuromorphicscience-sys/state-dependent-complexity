#!/usr/bin/env python3
"""Run code-release integrity, provenance and upload-readiness checks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/|/Users/)")
SECRET_PATTERNS = {
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def manifest_entries(data: Any) -> list[tuple[str, str]]:
    if isinstance(data, list):
        return [
            (item.get("path", ""), item.get("sha256", ""))
            for item in data if isinstance(item, dict)
        ]
    if isinstance(data, dict):
        files = data.get("files", data)
        if isinstance(files, dict):
            return [(str(path), str(digest)) for path, digest in files.items()]
        if isinstance(files, list):
            return manifest_entries(files)
    return []


def audit() -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    python_files = sorted(ROOT.rglob("*.py"))
    syntax_errors = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as error:  # pragma: no cover - diagnostic path
            syntax_errors.append({"path": relative(path), "error": repr(error)})
    if syntax_errors:
        failures.append(f"{len(syntax_errors)} Python syntax errors")

    cache_files = [
        relative(path) for path in ROOT.rglob("*")
        if (path.is_dir() and path.name == "__pycache__")
        or (path.is_file() and path.suffix.lower() in {".pyc", ".pyo"})
    ]
    if cache_files:
        failures.append(f"{len(cache_files)} Python cache artifacts")

    large_files = [
        {"path": relative(path), "bytes": path.stat().st_size}
        for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size > 1_000_000
    ]
    if large_files:
        warnings.append(f"{len(large_files)} files exceed 1 MB")

    hardcoded_paths = []
    secret_hits = []
    excluded = {relative(Path(__file__).resolve())}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".ps1", ".bat", ".sh", ".md", ".txt", ".json", ".csv"
        }:
            continue
        rel = relative(path)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if rel not in excluded and ABSOLUTE_PATH.search(line):
                hardcoded_paths.append({"path": rel, "line": number})
            if rel not in excluded:
                for label, pattern in SECRET_PATTERNS.items():
                    if pattern.search(line):
                        secret_hits.append({"path": rel, "line": number, "kind": label})
    if hardcoded_paths:
        warnings.append(
            f"{len(hardcoded_paths)} historical absolute-path references; use release_tools wrappers"
        )
    if secret_hits:
        failures.append(f"{len(secret_hits)} potential secret values")

    freeze_checked = 0
    freeze_mismatches = []
    for manifest in ROOT.rglob("FREEZE_MANIFEST.json"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for name, expected in data.get("sha256", {}).items():
            path = manifest.parent / name
            if path.suffix.lower() != ".py" or not path.exists():
                continue
            freeze_checked += 1
            actual = sha256(path)
            if actual != expected:
                freeze_mismatches.append(relative(path))
    if freeze_mismatches:
        failures.append(f"{len(freeze_mismatches)} frozen Python hash mismatches")

    package_checked = 0
    package_mismatches = []
    for name in ("package_manifest.json", "PACKAGE_MANIFEST_SHA256.json"):
        for manifest in ROOT.rglob(name):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for item, expected in manifest_entries(data):
                path = manifest.parent / item
                if not item or not expected or not path.is_file():
                    continue
                package_checked += 1
                if sha256(path) != expected:
                    package_mismatches.append(relative(path))
    if package_mismatches:
        failures.append(f"{len(package_mismatches)} package-manifest hash mismatches")

    stage4a_dir = ROOT / "analysis/model_stages/stage04a"
    stage4a_paths = [
        stage4a_dir / "formal_frozen/configs/stage4a.json",
        stage4a_dir / "archived_snapshot/configs__stage4a.json",
        stage4a_dir / "archived_snapshot/stage4a_config_snapshot.json",
    ]
    stage4a_configs = [json.loads(path.read_text(encoding="utf-8")) for path in stage4a_paths]
    stage4a_equal = all(config == stage4a_configs[0] for config in stage4a_configs[1:])
    expected_optimizer = {
        "population": 4096,
        "elite": 512,
        "generations": 6,
        "random_injection": 512,
        "swap_mutations": 2,
    }
    if not stage4a_equal or stage4a_configs[0].get("optimizer") != expected_optimizer:
        failures.append("Stage 4A formal/archive configuration mismatch")

    metadata_missing = [name for name in ("LICENSE", "CITATION.cff") if not (ROOT / name).exists()]
    if metadata_missing:
        warnings.append("release metadata still required: " + ", ".join(metadata_missing))
    license_verified = False
    if not metadata_missing:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        license_verified = (
            "GNU GENERAL PUBLIC LICENSE" in license_text
            and "Version 3, 29 June 2007" in license_text
            and "license: GPL-3.0-only" in citation_text
        )
        if not license_verified:
            failures.append("GPL-3.0-only license metadata is incomplete or inconsistent")
    lock_names = ("analysis-lock.txt", "qa-lock.txt", "full-lock.txt")
    environment_lock_missing = [
        name for name in lock_names if not (ROOT / "requirements" / name).exists()
    ]
    if environment_lock_missing:
        warnings.append(
            "exact dependency locks still required: " + ", ".join(environment_lock_missing)
        )

    checksum_manifest = ROOT / "SHA256SUMS"
    checksum_checked = 0
    checksum_mismatches: list[str] = []
    checksum_malformed: list[str] = []
    if not checksum_manifest.exists():
        warnings.append("release checksum manifest still required: SHA256SUMS")
    else:
        for number, line in enumerate(checksum_manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                expected, raw_path = line.split(maxsplit=1)
            except ValueError:
                checksum_malformed.append(f"line {number}")
                continue
            rel_path = raw_path.strip()
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            path = ROOT / rel_path
            if (
                len(expected) != 64
                or not all(char in "0123456789abcdef" for char in expected)
                or not rel_path
                or Path(rel_path).is_absolute()
                or rel_path in {"SHA256SUMS", "docs/release_audit.json"}
                or not path.is_file()
            ):
                checksum_malformed.append(f"line {number}")
                continue
            checksum_checked += 1
            if sha256(path) != expected:
                checksum_mismatches.append(rel_path)
    if checksum_malformed:
        failures.append(f"{len(checksum_malformed)} malformed release checksum entries")
    if checksum_mismatches:
        failures.append(f"{len(checksum_mismatches)} release checksum mismatches")

    return {
        "status": "PASS" if not failures else "FAIL",
        "upload_ready": (
            not failures
            and not metadata_missing
            and not environment_lock_missing
            and checksum_manifest.exists()
        ),
        "failures": failures,
        "warnings": warnings,
        "counts": {
            "python_files": len(python_files),
            "syntax_errors": len(syntax_errors),
            "cache_artifacts": len(cache_files),
            "large_files": len(large_files),
            "absolute_path_references": len(hardcoded_paths),
            "potential_secrets": len(secret_hits),
            "frozen_python_hashes_checked": freeze_checked,
            "frozen_python_hash_mismatches": len(freeze_mismatches),
            "package_files_checked": package_checked,
            "package_file_mismatches": len(package_mismatches),
            "release_checksum_files_checked": checksum_checked,
            "release_checksum_mismatches": len(checksum_mismatches),
            "release_checksum_malformed": len(checksum_malformed),
        },
        "stage4a": {
            "formal_archive_semantic_equal": stage4a_equal,
            "optimizer": stage4a_configs[0].get("optimizer"),
        },
        "metadata_missing": metadata_missing,
        "license_verified": license_verified,
        "environment_lock_missing": environment_lock_missing,
        "release_checksum_manifest": relative(checksum_manifest) if checksum_manifest.exists() else None,
        "details": {
            "syntax_errors": syntax_errors,
            "large_files": large_files,
            "hardcoded_paths": hardcoded_paths,
            "secret_hits": secret_hits,
            "freeze_mismatches": freeze_mismatches,
            "package_mismatches": package_mismatches,
            "checksum_mismatches": checksum_mismatches,
            "checksum_malformed": checksum_malformed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Also write the JSON report to this path.")
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        report = args.report if args.report.is_absolute() else ROOT / args.report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
