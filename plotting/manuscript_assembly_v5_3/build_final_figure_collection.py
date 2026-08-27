#!/usr/bin/env python3
"""Assemble the 20 frozen manuscript figures into one versioned collection."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT / "nature_figure_trial"
OUT = ROOT / "final_figure_collection_v5_3"
FIGURES = [
    *(f"Fig{i:02d}" for i in range(1, 7)),
    *(f"ED{i:02d}" for i in range(1, 7)),
    *(f"SI{i:02d}" for i in range(1, 9)),
]
FORMAT_DIR = {
    ".pdf": "PDF",
    ".png": "PNG_600dpi",
    ".svg": "SVG_EDITABLE",
    ".tiff": "TIFF_600dpi",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_freeze(figure_dir: Path, manifest: dict) -> None:
    if manifest.get("status") != "FROZEN":
        raise RuntimeError(f"Not frozen: {figure_dir.name}")
    for rel, expected in manifest["sha256"].items():
        src = (figure_dir / rel).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        actual = sha256(src)
        if actual != expected:
            raise RuntimeError(f"Freeze hash mismatch: {src}: {expected} != {actual}")


def select_artifact(release: Path, suffix: str) -> Path | None:
    hits = sorted(p for p in release.glob(f"*{suffix}") if p.is_file())
    if len(hits) > 1:
        hits = [p for p in hits if "full_nature" in p.name]
    if not hits:
        return None
    if len(hits) != 1:
        raise RuntimeError(f"Ambiguous {suffix} artifact in {release}: {hits}")
    return hits[0]


def main() -> None:
    for folder in [*FORMAT_DIR.values(), "metadata", "schematic_briefs"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    records = []
    collection = {"collection":"final_figure_collection_v5_3", "status":"FROZEN_REFERENCE_SET",
                  "figure_count":len(FIGURES), "figures":{}}
    for code in FIGURES:
        figure_dir = TRIAL / code
        freeze_path = figure_dir / "FREEZE_MANIFEST.json"
        manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
        validate_freeze(figure_dir, manifest)
        release = figure_dir / manifest["release_directory"]
        copied = {}
        for suffix, folder in FORMAT_DIR.items():
            src = select_artifact(release, suffix)
            if src is None:
                continue
            dst = OUT / folder / f"{code}{suffix}"
            shutil.copy2(src, dst)
            if suffix == ".pdf":
                doc = fitz.open(dst)
                if len(doc) != 1:
                    raise RuntimeError(f"Expected one-page figure PDF: {dst}")
                doc.close()
            elif suffix == ".png":
                with Image.open(dst) as im:
                    if min(im.size) < 1000:
                        raise RuntimeError(f"Unexpectedly small preview: {dst}: {im.size}")
            digest = sha256(dst)
            copied[suffix[1:]] = {"path":str(dst.relative_to(OUT)), "sha256":digest,
                                  "bytes":dst.stat().st_size,
                                  "source":str(src.relative_to(ROOT))}
            records.append((code,suffix[1:],str(dst.relative_to(OUT)),digest,
                            str(dst.stat().st_size),str(src.relative_to(ROOT))))

        source_manifest = release / "source_data_manifest.json"
        if source_manifest.exists():
            shutil.copy2(source_manifest, OUT / "metadata" / f"{code}_source_data_manifest.json")
        shutil.copy2(freeze_path, OUT / "metadata" / f"{code}_FREEZE_MANIFEST.json")
        collection["figures"][code] = {
            "release_directory":manifest["release_directory"],
            "freeze_note":manifest.get("note", ""),
            "artifacts":copied,
        }

    header="figure\tformat\tcollection_path\tsha256\tbytes\toriginal_source"
    lines=[header,*["\t".join(r) for r in records]]
    (OUT/"FILE_MANIFEST.tsv").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT/"COLLECTION_MANIFEST.json").write_text(
        json.dumps(collection,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    briefs={
        "Fig02a_controlled_resource_allocation.txt": ROOT.parent / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig02" / "Fig02_a__Controlled_resource-versus-allocation_design" / "00_SCHEMATIC_BRIEF.txt",
        "Fig03a_dynamical_vs_static.txt": ROOT.parent / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig03" / "Fig03_a__Dynamical_leverage_versus_static_prominence" / "00_SCHEMATIC_BRIEF.txt",
    }
    for name,src in briefs.items():
        shutil.copy2(src,OUT/"schematic_briefs"/name)

    expected={"pdf":20,"png":20,"svg":20,"tiff":20}
    actual={fmt:sum(1 for r in records if r[1]==fmt) for fmt in expected}
    if actual != expected:
        raise RuntimeError(f"Incomplete collection: expected {expected}, found {actual}")
    print(OUT)
    print(json.dumps(actual,sort_keys=True))


if __name__ == "__main__":
    main()
