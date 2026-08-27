from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import pandas as pd
import torch

def atomic_write_text(path: Path, text: str):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def atomic_write_json(path: Path, obj):
    atomic_write_text(Path(path), json.dumps(obj, indent=2, sort_keys=True))

def atomic_write_csv(path: Path, df: pd.DataFrame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def atomic_torch_save(path: Path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
