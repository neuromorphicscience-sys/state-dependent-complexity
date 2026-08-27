from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import pandas as pd


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def atomic_write_csv(path: Path, df: pd.DataFrame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def scan_completed_run_ids(chunks_dir: Path):
    done = set()
    for path in sorted(Path(chunks_dir).glob("batch_*.csv")):
        try:
            df = pd.read_csv(path, usecols=["run_id"])
            done.update(df["run_id"].astype(str).tolist())
        except Exception:
            continue
    return done


def consolidate_chunks(chunks_dir: Path, output_csv: Path):
    dfs = []
    for path in sorted(Path(chunks_dir).glob("batch_*.csv")):
        try:
            dfs.append(pd.read_csv(path))
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No valid Stage 3 chunk files found")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["run_id"], keep="last")
    atomic_write_csv(Path(output_csv), df)
    return df
