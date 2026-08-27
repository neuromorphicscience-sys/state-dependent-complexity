#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenScope Illusion 000248 — exact local NWB I/O v2.

This version uses the actual DANDI 000248 stimulus schema confirmed by:
1) the user's local NWB schema probe;
2) Allen Institute OpenScope Databook example code.

Critical correction from v1
---------------------------
Rows in e.g. `ICwcfg1_presentations` have stimulus_name == "ICwcfg1".
The actual image identity is encoded by the integer `frame`.

Allen OpenScope Databook mapping for cfg1/cfg0:
frame 3 -> IC1
frame 4 -> LC1
frame 6 -> LC2
frame 7 -> IC2
frame 8 -> IRE1
frame 9 -> IRE2
frame 10 -> TRE1
frame 11 -> TRE2

The full 0..21 descriptions are kept below for auditable mapping.

Only rows with the desired nonblank frames are treated as stimulus trials.
ITI rows (mostly frame 0) are never used as IC/LC trials.
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

TARGET_AREAS = ["V1", "LM", "RL", "AL", "PM", "AM"]

# Allen CCF / Visual Coding acronyms.
AREA_PATTERNS = [
    ("V1", [r"\bvisp", r"\bv1\b", r"primary visual"]),
    ("LM", [r"\bvisl", r"\blm\b", r"lateromedial"]),
    ("RL", [r"\bvisrl", r"\brl\b", r"rostrolateral"]),
    ("AL", [r"\bvisal", r"\bal\b", r"anterolateral"]),
    ("PM", [r"\bvispm", r"\bpm\b", r"posteromedial"]),
    ("AM", [r"\bvisam", r"\bam\b", r"anteromedial"]),
]

# Exact mapping reproduced from the Allen OpenScope Databook helper
# `get_stim_table_info`.  The last cfg1 label is normalized to OutTR; the
# Databook display contains a duplicated OutTL typo there, irrelevant to the
# present IC/LC/IRE/TRE analyses.
FRAME_LABELS_CFG1 = {
    0:"BLANK", 1:"X", 2:"TC1", 3:"IC1", 4:"LC1", 5:"TC2",
    6:"LC2", 7:"IC2", 8:"IRE1", 9:"IRE2", 10:"TRE1", 11:"TRE2",
    12:"XRE1", 13:"XRE2", 14:"InBR", 15:"InBL", 16:"InTL", 17:"InTR",
    18:"OutBR", 19:"OutBL", 20:"OutTL", 21:"OutTR",
}
FRAME_LABELS_CFG0 = {
    0:"BLANK", 1:"X", 2:"TC1", 3:"IC1", 4:"LC1", 5:"TC2",
    6:"LC2", 7:"IC2", 8:"IRE1", 9:"IRE2", 10:"TRE1", 11:"TRE2",
    12:"XRE1", 13:"XRE2", 14:"InR", 15:"InB", 16:"InL", 17:"InT",
    18:"OutR", 19:"OutB", 20:"OutL", 21:"OutT",
}

PRIMARY_LABELS = ["IC1","IC2","LC1","LC2"]
REAL_EDGE_LABELS = ["IRE1","IRE2","TRE1","TRE2"]
ANALYSIS_LABELS = PRIMARY_LABELS + REAL_EDGE_LABELS
IC_BLOCKS = ["ICwcfg1","ICwcfg0","ICkcfg1","ICkcfg0"]

WINDOWS = {
    "baseline": (-0.30, 0.00),
    "early": (0.05, 0.18),
    "late": (0.20, 0.40),
    "full": (0.00, 0.40),
}

def load_inventory(data_root: Path) -> pd.DataFrame:
    inv_path = data_root / "dandi_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        if "path" not in inv.columns:
            raise ValueError(f"{inv_path} lacks a path column")
    else:
        rows=[]
        for p in data_root.rglob("*.nwb"):
            rows.append({
                "asset_id":"",
                "path":str(p.relative_to(data_root)),
                "size":p.stat().st_size,
            })
        inv=pd.DataFrame(rows)
    if inv.empty:
        raise FileNotFoundError(f"No NWB files found under {data_root}")
    inv["local_path"]=inv["path"].map(lambda x:str((data_root/str(x)).resolve()))
    inv["exists"]=inv["local_path"].map(lambda x:Path(x).exists())
    inv["subject"]=inv["path"].str.extract(r"sub-(\d+)",expand=False)
    inv["session"]=inv["path"].str.extract(r"ses-(\d+)",expand=False)
    inv["probe"]=inv["path"].str.extract(r"probe-(\d+)",expand=False)
    inv["kind"]=np.where(
        inv["path"].str.contains(r"_ecephys\.nwb$",regex=True),"ecephys",
        np.where(inv["path"].str.contains(r"_ogen\.nwb$",regex=True),"ogen","other")
    )
    return inv

def build_session_manifest(inv: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (sub,ses),g in inv.groupby(["subject","session"],dropna=False):
        probes=g[g["kind"].eq("ecephys")].sort_values("probe")
        ogen=g[g["kind"].eq("ogen")].sort_values("path")
        rows.append({
            "subject":str(sub),
            "session":str(ses),
            "n_probe_files":int(len(probes)),
            "probe_files":json.dumps(probes["local_path"].tolist()),
            "ogen_files":json.dumps(ogen["local_path"].tolist()),
            "all_files_exist":bool(g["exists"].all()),
            "total_bytes":int(g["size"].sum()) if "size" in g.columns else np.nan,
            # DANDI 000248 stores the complete session Units table and spike_times
            # in the main *_ogen.nwb file. Per-probe *_ecephys.nwb assets are auxiliary
            # probe/LFP files and are NOT required for unit extraction.
            "eligible_ecephys":bool(len(ogen)>0 and ogen["exists"].all()),
        })
    return pd.DataFrame(rows).sort_values(["subject","session"]).reset_index(drop=True)

def canonical_area(value) -> Optional[str]:
    s=str(value).lower()
    s=re.sub(r"[_/\-]+"," ",s)
    for area,pats in AREA_PATTERNS:
        if any(re.search(p,s,flags=re.I) for p in pats):
            return area
    return None

def row_area(row: pd.Series) -> Optional[str]:
    preferred=[
        "location","structure_acronym","ecephys_structure_acronym",
        "structure","anatomical_location","brain_region","area"
    ]
    vals=[]
    for c in preferred:
        if c in row.index and pd.notna(row[c]):
            vals.append(row[c])
    if not vals:
        for c in row.index:
            if any(k in str(c).lower() for k in ["location","structure","area","region"]):
                if pd.notna(row[c]):
                    vals.append(row[c])
    for v in vals:
        a=canonical_area(v)
        if a:
            return a
    return None

def stimulus_group(label) -> Optional[str]:
    s=str(label)
    if s.startswith("IC"): return "IC"
    if s.startswith("LC"): return "LC"
    if s.startswith("IRE"): return "IRE"
    if s.startswith("TRE"): return "TRE"
    if s=="BLANK": return "BLANK"
    return "OTHER"

def frame_mapping(block: str) -> Dict[int,str]:
    return FRAME_LABELS_CFG1 if "cfg1" in block.lower() else FRAME_LABELS_CFG0

def decode_interval_table(df: pd.DataFrame, block: str) -> pd.DataFrame:
    d=df.copy()
    if "id" not in d.columns:
        d=d.reset_index(drop=False).rename(columns={"index":"id"})
    d["frame"]=pd.to_numeric(d["frame"],errors="coerce")
    mapping=frame_mapping(block)
    d["frame_int"]=d["frame"].round().astype("Int64")
    d["canonical_stimulus"]=d["frame_int"].map(mapping)
    d["stimulus_group"]=d["canonical_stimulus"].map(stimulus_group)
    d["stimulus_config"]=block
    d["orientation_config"]="cfg1" if "cfg1" in block.lower() else "cfg0"
    d["polarity"]="white" if block.lower().startswith("icw") else "black"
    d["is_primary_inference"]=d["canonical_stimulus"].isin(PRIMARY_LABELS)
    d["is_real_edge_control"]=d["canonical_stimulus"].isin(REAL_EDGE_LABELS)
    d["start_time"]=pd.to_numeric(d["start_time"],errors="coerce")
    d["stop_time"]=pd.to_numeric(d["stop_time"],errors="coerce")
    d["duration_s"]=d["stop_time"]-d["start_time"]
    return d

def extract_stimulus_trials(ogen_file: str):
    """
    Read the four IC interval tables and return:
    - analysis trials: only IC1/IC2/LC1/LC2/IRE1/IRE2/TRE1/TRE2 rows;
    - mapping audit across all 0..21 frames.
    """
    from pynwb import NWBHDF5IO
    trial_parts=[]
    audit_parts=[]
    block_meta=[]
    with NWBHDF5IO(ogen_file,"r",load_namespaces=True) as io:
        nwb=io.read()
        available=set(nwb.intervals.keys())
        for block in IC_BLOCKS:
            table_name=f"{block}_presentations"
            if table_name not in available:
                block_meta.append({"block":block,"status":"MISSING"})
                continue
            raw=nwb.intervals[table_name].to_dataframe().reset_index(drop=False)
            d=decode_interval_table(raw,block)

            cnt=(d.groupby(["stimulus_config","frame_int","canonical_stimulus"],
                           dropna=False)
                   .agg(n_rows=("frame_int","size"),
                        median_duration_s=("duration_s","median"),
                        first_start=("start_time","min"),
                        last_stop=("stop_time","max"))
                   .reset_index())
            audit_parts.append(cnt)

            use=d[d["canonical_stimulus"].isin(ANALYSIS_LABELS)].copy()
            use=use[use["start_time"].notna()].copy()
            keep_cols=[
                "id","start_time","stop_time","duration_s",
                "stimulus_name","stimulus_block","frame_int",
                "canonical_stimulus","stimulus_group","stimulus_config",
                "orientation_config","polarity",
                "is_primary_inference","is_real_edge_control",
            ]
            use=use[[c for c in keep_cols if c in use.columns]].copy()
            trial_parts.append(use)

            main_counts=use[use["canonical_stimulus"].isin(PRIMARY_LABELS)]["canonical_stimulus"].value_counts()
            block_meta.append({
                "block":block,
                "status":"OK",
                "n_interval_rows":int(len(d)),
                "n_analysis_rows":int(len(use)),
                "main_counts":{k:int(main_counts.get(k,0)) for k in PRIMARY_LABELS},
                "median_duration_s":float(use["duration_s"].median()) if len(use) else np.nan,
            })

        if not trial_parts:
            raise RuntimeError(f"No analyzable IC interval tables in {ogen_file}")

        trials=pd.concat(trial_parts,ignore_index=True)
        trials=trials.sort_values("start_time").reset_index(drop=True)
        trials["trial_uid"]=np.arange(len(trials),dtype=int)

        # Running speed is optional but valuable as a state-confound audit.
        running_meta={"status":"NOT_FOUND"}
        try:
            obj=None
            for _,candidate in nwb.objects.items():
                if str(getattr(candidate,"name","")).lower()=="running_speed":
                    obj=candidate
                    break
            if obj is not None:
                data=np.asarray(obj.data[:],dtype=float).squeeze()
                if getattr(obj,"timestamps",None) is not None:
                    ts=np.asarray(obj.timestamps[:],dtype=float).squeeze()
                elif getattr(obj,"rate",None) not in [None,0]:
                    ts=float(obj.starting_time)+np.arange(len(data))/float(obj.rate)
                else:
                    ts=None
                if ts is not None and len(ts)==len(data):
                    pre=[]
                    stim=[]
                    for onset in trials["start_time"].to_numpy(float):
                        a=np.searchsorted(ts,onset-0.30,"left")
                        b=np.searchsorted(ts,onset,"left")
                        c=np.searchsorted(ts,onset+0.40,"left")
                        pre.append(float(np.nanmean(data[a:b])) if b>a else np.nan)
                        stim.append(float(np.nanmean(data[b:c])) if c>b else np.nan)
                    trials["pre_running_speed"]=pre
                    trials["stim_running_speed"]=stim
                    running_meta={
                        "status":"OK",
                        "n_samples":int(len(data)),
                        "finite_pretrial_fraction":float(np.mean(np.isfinite(pre))),
                    }
        except Exception as e:
            running_meta={"status":"ERROR","error":repr(e)}

    audit=pd.concat(audit_parts,ignore_index=True) if audit_parts else pd.DataFrame()
    meta={
        "ogen_file":ogen_file,
        "blocks":block_meta,
        "running":running_meta,
        "mapping_source":"Allen OpenScope Databook get_stim_table_info trialtypedescription",
        "primary_block":"ICwcfg1",
        "primary_frame_mapping":{"3":"IC1","4":"LC1","6":"LC2","7":"IC2"},
        "real_edge_frame_mapping":{"8":"IRE1","9":"IRE2","10":"TRE1","11":"TRE2"},
    }

    # Hard guardrails for primary block.
    p=trials[trials["stimulus_config"].eq("ICwcfg1")]
    counts=p[p["canonical_stimulus"].isin(PRIMARY_LABELS)]["canonical_stimulus"].value_counts()
    missing=[x for x in PRIMARY_LABELS if int(counts.get(x,0))<200]
    if missing:
        raise RuntimeError(
            f"Primary frame mapping failed/underfilled for {missing}; counts={counts.to_dict()}"
        )
    med=float(p[p["canonical_stimulus"].isin(PRIMARY_LABELS)]["duration_s"].median())
    if not (0.30 <= med <= 0.50):
        raise RuntimeError(f"Unexpected primary stimulus duration median={med}")

    return trials,audit,meta

def _paper_good_mask(units: pd.DataFrame) -> pd.Series:
    """
    Main quality criterion:
    if a quality column exists, retain 'good' single units.
    """
    m=pd.Series(True,index=units.index)
    if "quality" in units.columns:
        q=units["quality"].astype(str).str.lower()
        informative=q.isin(["good","noise","mua"]).any()
        if informative:
            m &= q.eq("good")
    return m

def _strict_qc_mask(units: pd.DataFrame) -> pd.Series:
    m=_paper_good_mask(units)
    rules=[
        ("presence_ratio",">=",0.90),
        ("amplitude_cutoff","<=",0.10),
        ("isi_violations","<=",0.50),
        ("isi_violations_ratio","<=",0.50),
    ]
    for c,op,thr in rules:
        if c not in units.columns:
            continue
        x=pd.to_numeric(units[c],errors="coerce")
        finite=x.notna()
        if finite.sum()<max(10,len(units)//5):
            continue
        m &= (~finite) | ((x>=thr) if op==">=" else (x<=thr))
    return m

def _waveform_ttp_ms(value, sampling_rate_hz=30000.0):
    try:
        a=np.asarray(value,dtype=float)
    except Exception:
        return np.nan
    if a.size<5:
        return np.nan
    a=np.squeeze(a)
    if a.ndim==1:
        w=a
    else:
        # Treat final axis as time; choose the trace with largest peak-to-peak.
        b=a.reshape(-1,a.shape[-1])
        ptp=np.nanmax(b,axis=1)-np.nanmin(b,axis=1)
        if not np.isfinite(ptp).any():
            return np.nan
        w=b[int(np.nanargmax(ptp))]
    if len(w)<5 or not np.isfinite(w).any():
        return np.nan
    trough=int(np.nanargmin(w))
    if trough>=len(w)-2:
        return np.nan
    post=w[trough+1:]
    if not np.isfinite(post).any():
        return np.nan
    peak=trough+1+int(np.nanargmax(post))
    return float((peak-trough)/sampling_rate_hz*1000.0)

def _regular_spiking(units: pd.DataFrame):
    """
    RS = extracellular trough-to-peak width >= 0.4 ms.

    First use a scalar width column if present. Otherwise compute from
    waveform_mean at the Neuropixels 30-kHz spike-band sampling rate.
    """
    candidates=[
        "waveform_duration_ms","trough_to_peak_ms",
        "waveform_duration","trough_to_peak","trough_to_peak_duration",
        "waveform_width","spike_width",
    ]
    for c in candidates:
        if c not in units.columns:
            continue
        x=pd.to_numeric(units[c],errors="coerce")
        finite=x[np.isfinite(x)]
        if len(finite)<max(10,len(units)//10):
            continue
        med=float(np.nanmedian(finite))
        lc=c.lower()
        if lc.endswith("_ms"):
            ms=x
            unit="ms"
        elif med<0.02:
            ms=x*1000.0
            unit="seconds_inferred"
        elif med<10:
            ms=x
            unit="ms_inferred"
        else:
            continue
        return (ms>=0.4).fillna(False), ms, {
            "resolved":True,"method":f"scalar:{c}","input_unit":unit,
            "threshold_ms":0.4,
        }

    if "waveform_mean" in units.columns:
        ms=units["waveform_mean"].map(_waveform_ttp_ms)
        if ms.notna().sum()>=max(10,len(units)//10):
            return (ms>=0.4).fillna(False), ms, {
                "resolved":True,"method":"waveform_mean_30kHz",
                "input_unit":"samples_at_30kHz","threshold_ms":0.4,
            }

    return pd.Series(True,index=units.index), pd.Series(np.nan,index=units.index), {
        "resolved":False,"method":"unresolved","input_unit":None,"threshold_ms":0.4,
    }

def extract_units_from_session_nwb(fp: str, subject: str, session: str):
    """
    Extract all sorted units/spike_times from the main *_ogen.nwb session file.

    IMPORTANT DANDI 000248 schema fact
    ----------------------------------
    The main *_ogen.nwb contains:
      - the complete Units table;
      - spike_times;
      - quality metrics;
      - waveform_mean;
      - peak_channel_id;
      - the full electrodes table for all six Neuropixels probes.

    The per-probe *_ecephys.nwb files do not carry the sorted Units table used
    by the OpenScope Databook analysis, so v2.1 intentionally does not extract
    units from those files.
    """
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(fp, "r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.units is None:
            raise RuntimeError(f"Main session NWB has no Units table: {fp}")

        udf = nwb.units.to_dataframe()
        if len(udf) == 0:
            raise RuntimeError(f"Main session Units table is empty: {fp}")
        if "spike_times" not in udf.columns:
            raise RuntimeError(
                f"Main session Units table has no spike_times; columns={list(udf.columns)}"
            )

        spikes = [np.asarray(x, dtype=np.float64) for x in udf["spike_times"]]

        # Keep the original unit id before resetting the DataFrame index.
        original_id = udf.index.to_numpy()
        meta = udf.drop(columns=["spike_times"]).copy()
        meta["local_unit_id"] = original_id.astype(str)
        meta = meta.reset_index(drop=True)

        meta["subject"] = str(subject)
        meta["session"] = str(session)
        meta["source_file"] = fp
        meta["unit_uid"] = [
            f"sub-{subject}_ses-{session}_unit-{u}"
            for u in meta["local_unit_id"]
        ]

        # Official OpenScope Databook mapping:
        # unit.peak_channel_id -> electrodes.id -> electrodes.location.
        channel_location = {}
        channel_probe = {}
        if nwb.electrodes is not None:
            edf = nwb.electrodes.to_dataframe()
            # In PyNWB the electrode id is normally the dataframe index.
            for eid, row in edf.iterrows():
                channel_location[eid] = row.get("location", None)
                # Depending on PyNWB version, group may be an object and
                # group_name may be available as a scalar column.
                gname = row.get("group_name", None)
                if gname is None:
                    gobj = row.get("group", None)
                    gname = getattr(gobj, "name", None) if gobj is not None else None
                channel_probe[eid] = gname

        if "peak_channel_id" in meta.columns:
            peak = pd.to_numeric(meta["peak_channel_id"], errors="coerce")
            def _lookup(mapping, value):
                if not np.isfinite(value):
                    return None
                # IDs are integer-valued in this dataset.
                key = int(value)
                if key in mapping:
                    return mapping[key]
                # Defensive fallback for numpy scalar / string-like indexing.
                return mapping.get(value, None)
            meta["peak_channel_location"] = [
                _lookup(channel_location, x) for x in peak.to_numpy(float)
            ]
            meta["probe_name"] = [
                _lookup(channel_probe, x) for x in peak.to_numpy(float)
            ]
        else:
            meta["peak_channel_location"] = None
            meta["probe_name"] = None

        # Prefer the electrode-mapped location because the Units table itself
        # does not necessarily include a location column.
        meta["area"] = meta["peak_channel_location"].map(canonical_area)

        # Fallback to any unit-level anatomy columns if needed.
        miss = meta["area"].isna()
        if miss.any():
            meta.loc[miss, "area"] = meta.loc[miss].apply(row_area, axis=1)

        meta["target_area"] = meta["area"].isin(TARGET_AREAS)
        meta["paper_good"] = _paper_good_mask(meta).to_numpy(bool)
        meta["strict_qc"] = _strict_qc_mask(meta).to_numpy(bool)

        rs, ttp, rsmeta = _regular_spiking(meta)
        meta["regular_spiking"] = rs.to_numpy(bool)
        meta["trough_to_peak_ms"] = pd.to_numeric(ttp, errors="coerce")
        meta["rs_resolved"] = bool(rsmeta["resolved"])
        meta["rs_method"] = rsmeta["method"]

        # Drop large array-valued metadata after RS derivation. spike_times are
        # already stored separately in `spikes`.
        drop_large = []
        for c in meta.columns:
            if c in {
                "unit_uid", "local_unit_id", "quality", "peak_channel_id",
                "peak_channel_location", "probe_name", "area"
            }:
                continue
            if meta[c].dtype == object:
                vals = meta[c].dropna().head(8).tolist()
                if any(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
                    drop_large.append(c)
        meta = meta.drop(columns=drop_large, errors="ignore")

        audit = {
            "n_units_total": int(len(meta)),
            "n_good": int(meta["paper_good"].sum()),
            "n_target_area": int(meta["target_area"].sum()),
            "n_target_good": int((meta["target_area"] & meta["paper_good"]).sum()),
            "rs_resolved": bool(rsmeta["resolved"]),
            "rs_method": rsmeta["method"],
            "n_regular_spiking": int(meta["regular_spiking"].sum()) if rsmeta["resolved"] else None,
            "area_counts_all": {
                str(k): int(v) for k, v in meta["area"].fillna("UNMAPPED").value_counts().items()
            },
            "area_counts_good": {
                str(k): int(v) for k, v in meta.loc[meta["paper_good"], "area"].fillna("UNMAPPED").value_counts().items()
            },
        }
        return meta, spikes, audit

def count_spikes_matrix(spikes: List[np.ndarray], onsets: np.ndarray,
                        window: Tuple[float,float]) -> np.ndarray:
    a,b=window
    starts=onsets+float(a)
    stops=onsets+float(b)
    out=np.empty((len(onsets),len(spikes)),dtype=np.float32)
    for j,st in enumerate(spikes):
        left=np.searchsorted(st,starts,side="left")
        right=np.searchsorted(st,stops,side="left")
        out[:,j]=(right-left).astype(np.float32)
    return out

def extract_session(row: pd.Series, cache_dir: Path, *, force=False,
                    windows=WINDOWS):
    subject=str(row["subject"])
    session=str(row["session"])
    sdir=cache_dir/f"sub-{subject}_ses-{session}"
    sdir.mkdir(parents=True,exist_ok=True)
    done=sdir/"done.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))

    status={"subject":subject,"session":session,"status":"STARTED"}
    try:
        probe_files=json.loads(row["probe_files"])
        ogen_files=json.loads(row["ogen_files"])
        if not ogen_files:
            raise RuntimeError("No OGEN file for stimulus timing")
        ogen=ogen_files[0]

        trials,audit,trial_meta=extract_stimulus_trials(ogen)
        trials.to_csv(sdir/"trials.csv",index=False)
        audit.to_csv(sdir/"stimulus_mapping_audit.csv",index=False)
        (sdir/"trial_mapping_meta.json").write_text(
            json.dumps(trial_meta,ensure_ascii=False,indent=2),encoding="utf-8"
        )

        # The main *_ogen.nwb is the complete session NWB containing all sorted
        # units and spike_times for the six probes.
        units, spikes_all, unit_audit = extract_units_from_session_nwb(
            ogen, subject, session
        )
        if len(units) != len(spikes_all):
            raise RuntimeError("unit metadata / spike list mismatch")

        (sdir/"unit_extraction_audit.json").write_text(
            json.dumps(unit_audit, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        rs_resolved=bool(units["rs_resolved"].any())
        primary_keep=units["target_area"] & units["paper_good"]
        if rs_resolved:
            primary_keep &= units["regular_spiking"]

        # If a session genuinely has too few paper-matched units, do not silently
        # change the primary definition. Keep a separate fallback cohort only for
        # diagnostics; the session is marked underpowered if primary < 40.
        strict_keep=units["target_area"] & units["strict_qc"]
        if rs_resolved:
            strict_keep &= units["regular_spiking"]

        units["primary_unit"]=primary_keep
        units["strict_unit"]=strict_keep
        units.to_csv(sdir/"units_all.csv",index=False)

        primary_idx=np.flatnonzero(primary_keep.to_numpy())
        units_primary=units.iloc[primary_idx].reset_index(drop=True)
        spikes_primary=[spikes_all[i] for i in primary_idx]
        units_primary.to_csv(sdir/"units_primary.csv",index=False)

        if len(units_primary)<20:
            raise RuntimeError(f"Too few primary visual RS/good units: {len(units_primary)}")

        onsets=trials["start_time"].to_numpy(float)
        mats={}
        for name,win in windows.items():
            mats[name]=count_spikes_matrix(spikes_primary,onsets,win)
        # Store IDs with explicit fixed-width Unicode dtype so future caches
        # remain readable with allow_pickle=False. Scientific matrices are
        # numeric float32 arrays.
        unit_ids = units_primary["unit_uid"].astype(str).to_numpy(dtype="U")
        np.savez_compressed(
            sdir/"spike_counts.npz",
            **mats,
            trial_uid=trials["trial_uid"].to_numpy(dtype=np.int64),
            unit_uid=unit_ids,
        )

        primary_counts=(
            trials[trials["stimulus_config"].eq("ICwcfg1") &
                   trials["canonical_stimulus"].isin(PRIMARY_LABELS)]
            ["canonical_stimulus"].value_counts().to_dict()
        )
        status.update({
            "status":"COMPLETE",
            "n_analysis_trials":int(len(trials)),
            "n_primary_ICLC_trials":int(sum(primary_counts.values())),
            "primary_label_counts":{k:int(primary_counts.get(k,0)) for k in PRIMARY_LABELS},
            "n_units_all":int(len(units)),
            "n_units_primary":int(len(units_primary)),
            "areas":units_primary["area"].value_counts().to_dict(),
            "unit_source":"main_ogen_nwb",
            "unit_audit":unit_audit,
            "rs_filter_resolved":rs_resolved,
            "rs_methods":sorted(units.loc[units["rs_resolved"],"rs_method"].dropna().astype(str).unique().tolist()),
            "running_status":trial_meta.get("running",{}),
            "cache_dir":str(sdir),
        })
    except Exception as e:
        status.update({
            "status":"FAILED",
            "error":repr(e),
            "traceback":traceback.format_exc(),
            "cache_dir":str(sdir),
        })
    done.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    return status
