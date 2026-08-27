#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

FIG_EXTS = {".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg", ".jpeg", ".eps"}
REPORT_EXTS = {".md", ".txt", ".json"}
CATALOG_EXTS = {".csv", ".tsv"}

# Folders we never want to recurse into when scanning.
SKIP_DIR_NAMES = {
    ".git", ".idea", ".venv", "venv", "__pycache__",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "AllStages_Figure_Atlas_v1",
}

# Strongly preferred plotting/result roots. These are scanned first.
PREFERRED_ROOT_HINTS = [
    "plot/publication_figures",
    "plot/Stage4_integrated_full",
    "plot/Stage5A_robustness_v1",
    "plot/Stage5BR_leverage_closure_v1",
    "plot/Stage5B",
    "plot/Stage5A",
    "plot/Stage4A",
    "plot/Stage4",
    "plot/Stage3",
    "plot/Stage2",
    "plot/Stage1B",
    "plot/Stage1",
]

STAGE_PATTERNS = [
    ("Stage5B", re.compile(r"stage[\s_\-]*5\s*b|stage5br|5b[\s_\-]*r", re.I)),
    ("Stage5A", re.compile(r"stage[\s_\-]*5\s*a", re.I)),
    ("Stage4",  re.compile(r"stage[\s_\-]*4|4a|4b|4c|4d|4e|4f", re.I)),
    ("Stage3",  re.compile(r"stage[\s_\-]*3", re.I)),
    ("Stage2",  re.compile(r"stage[\s_\-]*2", re.I)),
    ("Stage1B", re.compile(r"stage[\s_\-]*1\s*b|stage1b", re.I)),
    ("Stage1",  re.compile(r"stage[\s_\-]*1", re.I)),
]

# For generic publication_figures folders where stage is encoded in a parent
# such as earlier_stages, we use filename hints too.
FILENAME_STAGE_HINTS = [
    ("Stage5B", re.compile(r"(?:^|[_\-])(?:s?5b|5br)(?:[_\-]|$)", re.I)),
    ("Stage5A", re.compile(r"(?:^|[_\-])(?:s?5a)(?:[_\-]|$)", re.I)),
    ("Stage4",  re.compile(r"(?:^|[_\-])(?:s?4[a-f]?|stage4)(?:[_\-]|$)", re.I)),
    ("Stage3",  re.compile(r"(?:^|[_\-])(?:s?3|stage3)(?:[_\-]|$)", re.I)),
    ("Stage2",  re.compile(r"(?:^|[_\-])(?:s?2|stage2)(?:[_\-]|$)", re.I)),
    ("Stage1B", re.compile(r"(?:^|[_\-])(?:s?1b|stage1b)(?:[_\-]|$)", re.I)),
    ("Stage1",  re.compile(r"(?:^|[_\-])(?:s?1|stage1)(?:[_\-]|$)", re.I)),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect all Stage 1–5 figure outputs into one upload-ready atlas."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Default: current working directory.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/plot/AllStages_Figure_Atlas_v1",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Also create a .zip beside the output folder.",
    )
    p.add_argument(
        "--no-support",
        action="store_true",
        help="Do not copy lightweight reports/catalogs.",
    )
    p.add_argument(
        "--include-all-project-figures",
        action="store_true",
        help=(
            "Broaden scan to the entire project root. By default the scan prioritizes "
            "the plot tree and known result folders to avoid unrelated images."
        ),
    )
    p.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate image files instead of deduplicating by SHA256.",
    )
    return p.parse_args()

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())

def is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)

def infer_stage(path: Path, project_root: Path) -> str:
    rel = safe_rel(path, project_root).replace("\\", "/")
    # Parent/path-based rules first.
    for stage, pat in STAGE_PATTERNS:
        if pat.search(rel):
            return stage

    # Filename-based fallback.
    name = path.name
    for stage, pat in FILENAME_STAGE_HINTS:
        if pat.search(name):
            return stage

    # Some historical pipeline output may sit under publication_figures/earlier_stages.
    lower = rel.lower()
    if "earlier_stages" in lower:
        # Try extracting stage tokens from the filename.
        m = re.search(r"stage[_\- ]?(1b|1|2|3)", name, re.I)
        if m:
            token = m.group(1).upper()
            return "Stage1B" if token == "1B" else f"Stage{token}"

    return "Unclassified"

def classify_figure_role(path: Path) -> str:
    s = str(path).lower()
    if any(k in s for k in ("diagnostic", "audit", "inventory", "qc", "debug")):
        return "DIAGNOSTIC"
    if any(k in s for k in ("supp", "supplement", "si_", "figures_si", "extended", "ed_")):
        return "SUPPORTING"
    if any(k in s for k in ("main", "figures_main", "publication")):
        return "MAIN_CANDIDATE"
    return "UNSPECIFIED"

def png_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if path.suffix.lower() != ".png":
        return None, None
    try:
        from PIL import Image  # optional dependency
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None, None

def iter_files_under(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and not is_skipped(p):
            yield p

def candidate_scan_roots(project_root: Path, include_all: bool) -> List[Path]:
    roots: List[Path] = []
    seen = set()

    def add(p: Path):
        rp = str(p.resolve()).lower() if p.exists() else str(p).lower()
        if p.exists() and rp not in seen:
            seen.add(rp)
            roots.append(p)

    for hint in PREFERRED_ROOT_HINTS:
        add(project_root / Path(hint))

    # Always include the full plot tree, because earlier pipelines may have custom names.
    add(project_root / "plot")

    if include_all:
        add(project_root)

    return roots

def collect_candidates(project_root: Path, include_all: bool) -> List[Path]:
    roots = candidate_scan_roots(project_root, include_all)
    files: Dict[str, Path] = {}
    for r in roots:
        for p in iter_files_under(r):
            if p.suffix.lower() in FIG_EXTS:
                key = str(p.resolve()).lower()
                files[key] = p
    return sorted(files.values(), key=lambda x: str(x).lower())

def support_candidates(project_root: Path) -> List[Path]:
    plot_root = project_root / "plot"
    if not plot_root.exists():
        return []
    out = []
    for p in iter_files_under(plot_root):
        low = p.name.lower()
        ext = p.suffix.lower()
        if ext in REPORT_EXTS and (
            "report" in low
            or "summary" in low
            or "readme" in low
            or "metrics" in low
            or "status" in low
        ):
            out.append(p)
        elif ext in CATALOG_EXTS and (
            "catalog" in low
            or "summary" in low
            or "metrics" in low
        ):
            out.append(p)
    return sorted(set(out), key=lambda x: str(x).lower())

def unique_target_name(stage: str, src: Path, digest: str, used_names: set) -> str:
    stem = re.sub(r"[^A-Za-z0-9._\-]+", "_", src.stem).strip("_")
    ext = src.suffix.lower()
    base = f"{stage}__{stem}{ext}"
    if base.lower() not in used_names:
        used_names.add(base.lower())
        return base
    base = f"{stage}__{stem}__{digest[:8]}{ext}"
    used_names.add(base.lower())
    return base

def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def copy_support(files: List[Path], project_root: Path, out_root: Path) -> int:
    count = 0
    for src in files:
        stage = infer_stage(src, project_root)
        rel = safe_rel(src, project_root).replace("\\", "__").replace("/", "__")
        rel = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", rel)
        if src.suffix.lower() in CATALOG_EXTS:
            dest_dir = out_root / "support" / "catalogs" / stage
        else:
            dest_dir = out_root / "support" / "reports" / stage
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / rel
        try:
            shutil.copy2(src, dest)
            count += 1
        except Exception as e:
            print(f"[WARN] Support copy failed: {src} -> {e}")
    return count

def zip_folder(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(folder.parent)
                zf.write(p, arcname)

def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        print(f"[ERROR] Project root does not exist: {root}")
        return 2

    out_root = (args.out.resolve() if args.out else (root / "plot" / "AllStages_Figure_Atlas_v1"))
    # Clean previous output for a deterministic bundle.
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    figures_by_stage = out_root / "figures_by_stage"
    figures_flat = out_root / "figures_flat"
    figures_by_stage.mkdir(parents=True, exist_ok=True)
    figures_flat.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("All-stages figure atlas")
    print("=" * 88)
    print(f"Project root : {root}")
    print(f"Output root  : {out_root}")
    print()

    candidates = collect_candidates(root, args.include_all_project_figures)
    print(f"[1/5] Found {len(candidates)} candidate figure files.")

    records: List[dict] = []
    digest_to_primary: Dict[str, Path] = {}
    used_flat_names: set = set()
    stage_counts = Counter()
    duplicate_count = 0

    for idx, src in enumerate(candidates, start=1):
        try:
            digest = sha256_file(src)
        except Exception as e:
            print(f"[WARN] SHA256 failed: {src} -> {e}")
            continue

        stage = infer_stage(src, root)
        role = classify_figure_role(src)
        stage_dir = figures_by_stage / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        is_dup = digest in digest_to_primary
        if is_dup and not args.keep_duplicates:
            duplicate_count += 1
            primary = digest_to_primary[digest]
            records.append({
                "stage": stage,
                "role_hint": role,
                "source_path": str(src),
                "source_rel": safe_rel(src, root),
                "copied_stage_path": "",
                "copied_flat_path": "",
                "suffix": src.suffix.lower(),
                "bytes": src.stat().st_size,
                "width_px": "",
                "height_px": "",
                "sha256": digest,
                "duplicate": "YES",
                "duplicate_of": str(primary),
                "mtime": datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
            })
            continue

        digest_to_primary[digest] = src

        # Preserve original filename inside each stage folder; if collision occurs, add hash.
        stage_dest = stage_dir / src.name
        if stage_dest.exists():
            stage_dest = stage_dir / f"{src.stem}__{digest[:8]}{src.suffix.lower()}"

        flat_name = unique_target_name(stage, src, digest, used_flat_names)
        flat_dest = figures_flat / flat_name

        try:
            shutil.copy2(src, stage_dest)
            shutil.copy2(src, flat_dest)
        except Exception as e:
            print(f"[WARN] Copy failed: {src} -> {e}")
            continue

        w, h = png_dimensions(src)
        stage_counts[stage] += 1

        records.append({
            "stage": stage,
            "role_hint": role,
            "source_path": str(src),
            "source_rel": safe_rel(src, root),
            "copied_stage_path": safe_rel(stage_dest, out_root),
            "copied_flat_path": safe_rel(flat_dest, out_root),
            "suffix": src.suffix.lower(),
            "bytes": src.stat().st_size,
            "width_px": w if w is not None else "",
            "height_px": h if h is not None else "",
            "sha256": digest,
            "duplicate": "NO",
            "duplicate_of": "",
            "mtime": datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
        })

        if idx % 50 == 0:
            print(f"      processed {idx}/{len(candidates)}")

    print(f"[2/5] Copied {sum(stage_counts.values())} unique figure files.")
    print(f"      exact duplicates skipped: {duplicate_count}")

    support_count = 0
    if not args.no_support:
        support_files = support_candidates(root)
        support_count = copy_support(support_files, root, out_root)
    print(f"[3/5] Copied {support_count} lightweight report/catalog files.")

    fieldnames = [
        "stage", "role_hint", "source_path", "source_rel",
        "copied_stage_path", "copied_flat_path",
        "suffix", "bytes", "width_px", "height_px",
        "sha256", "duplicate", "duplicate_of", "mtime",
    ]
    write_csv(out_root / "figure_catalog.csv", records, fieldnames)

    with (out_root / "figure_catalog.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    stage_order = ["Stage1", "Stage1B", "Stage2", "Stage3", "Stage4", "Stage5A", "Stage5B", "Unclassified"]
    summary_rows = []
    for stage in stage_order:
        unique_n = stage_counts.get(stage, 0)
        total_records = sum(1 for r in records if r["stage"] == stage)
        dup_n = sum(1 for r in records if r["stage"] == stage and r["duplicate"] == "YES")
        if unique_n or total_records:
            summary_rows.append({
                "stage": stage,
                "unique_figures_copied": unique_n,
                "catalog_records": total_records,
                "duplicates_skipped": dup_n,
            })
    write_csv(
        out_root / "stage_summary.csv",
        summary_rows,
        ["stage", "unique_figures_copied", "catalog_records", "duplicates_skipped"],
    )

    # Provenance manifest.
    with (out_root / "source_manifest.txt").open("w", encoding="utf-8") as f:
        f.write(f"Created: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Project root: {root}\n")
        f.write(f"Unique figures copied: {sum(stage_counts.values())}\n")
        f.write(f"Duplicates skipped: {duplicate_count}\n")
        f.write(f"Support files copied: {support_count}\n\n")
        for r in records:
            f.write(
                f"[{r['stage']}] {r['source_rel']} | sha256={r['sha256']} "
                f"| duplicate={r['duplicate']}\n"
            )

    readme = f"""# All Stages Figure Atlas v1

Created: {datetime.now().isoformat(timespec='seconds')}

This bundle is intended for manuscript-level review of the complete Neural Science project.

## Contents

- `figures_by_stage/`
  - Stage1
  - Stage1B
  - Stage2
  - Stage3
  - Stage4
  - Stage5A
  - Stage5B
  - Unclassified
- `figures_flat/`
  - one flat directory for rapid visual review / upload
- `figure_catalog.csv`
  - source provenance, inferred stage, file type, hash, optional PNG dimensions
- `stage_summary.csv`
  - figure counts by stage
- `support/`
  - lightweight reports and catalogs only; no raw scientific data
- `source_manifest.txt`
  - provenance manifest

## Collection summary

Unique figures copied: {sum(stage_counts.values())}

Exact duplicate figures skipped: {duplicate_count}

Support files copied: {support_count}

## Notes

1. The collector does **not** recompute or redraw any result.
2. Exact duplicates are identified by SHA256 and skipped by default.
3. Stage assignment is inferred from folder and filename conventions.
4. `Unclassified/` should be checked manually before final manuscript triage.
5. The output is meant for figure selection into MAIN / Extended Data / SI / Diagnostic / Drop.
"""
    (out_root / "00_README.md").write_text(readme, encoding="utf-8")

    print("[4/5] Wrote catalogs and provenance manifest.")

    zip_path = out_root.with_suffix(".zip")
    if args.zip:
        print(f"[5/5] Creating zip: {zip_path.name}")
        zip_folder(out_root, zip_path)
    else:
        print("[5/5] ZIP not requested. Use --zip to create one.")

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    for row in summary_rows:
        print(
            f"{row['stage']:12s}  unique={row['unique_figures_copied']:4d}  "
            f"records={row['catalog_records']:4d}  dup={row['duplicates_skipped']:4d}"
        )
    print()
    print(f"Atlas folder : {out_root}")
    if args.zip:
        print(f"Upload ZIP   : {zip_path}")
    print()
    print("Please upload the ZIP (or the atlas folder contents) for manuscript-level figure triage.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
