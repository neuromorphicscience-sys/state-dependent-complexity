#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Synapse-model targeted audit v1
==========================================

Purpose:
After pair-level endpoint models show that raw STP/strength effects are largely
absorbed by identity/structure controls, inspect the *synapse_model* table
directly so that the next leverage definition can be based on standardized
simulated train responses rather than raw cell-level sums or isolated STP
summary endpoints.

Read-only. Does not scan pulse_response.
"""

from __future__ import annotations
import argparse, csv, json, os, sqlite3
from pathlib import Path
from urllib.parse import quote
import pandas as pd

DEFAULT_ROOT = Path.cwd()
DEFAULT_DB = DEFAULT_ROOT / "synphys_r2.1_full.sqlite"
DEFAULT_TRANSFER = DEFAULT_ROOT / "stage5c_synphys_transfer_v2" / "synphys_frozen_complexity_transfer.csv"
DEFAULT_OUT = DEFAULT_ROOT / "stage5c_synapse_model_audit_v1"

def qident(x):
    return '"' + x.replace('"','""') + '"'

def connect_ro(path):
    uri = f"file:{quote(str(path).replace(os.sep,'/'))}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=60000")
    return con

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=DEFAULT_DB)
    ap.add_argument("--transfer",type=Path,default=DEFAULT_TRANSFER)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--sample",type=int,default=20)
    args=ap.parse_args()

    db=args.db.resolve(); transfer=args.transfer.resolve(); out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not db.exists(): raise SystemExit(f"Missing DB: {db}")
    if not transfer.exists(): raise SystemExit(f"Missing transfer file: {transfer}")

    con=connect_ro(db)

    # Exact schema
    cols=con.execute('PRAGMA table_info("synapse_model")').fetchall()
    fks=con.execute('PRAGMA foreign_key_list("synapse_model")').fetchall()
    idx=con.execute('PRAGMA index_list("synapse_model")').fetchall()
    create_sql=con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='synapse_model'"
    ).fetchone()

    col_df=pd.DataFrame(cols,columns=["cid","name","type","notnull","default","pk"])
    fk_df=pd.DataFrame(fks,columns=["id","seq","ref_table","from_col","to_col","on_update","on_delete","match"])
    idx_df=pd.DataFrame(idx,columns=["seq","name","unique","origin","partial"])

    col_df.to_csv(out/"synapse_model_columns.csv",index=False,encoding="utf-8-sig")
    fk_df.to_csv(out/"synapse_model_foreign_keys.csv",index=False,encoding="utf-8-sig")
    idx_df.to_csv(out/"synapse_model_indexes.csv",index=False,encoding="utf-8-sig")
    (out/"synapse_model_create.sql").write_text((create_sql[0] if create_sql and create_sql[0] else "")+";\n",encoding="utf-8")

    # Row count is expected to be small relative to raw response tables; exact count is useful here.
    n_model=con.execute("SELECT COUNT(*) FROM synapse_model").fetchone()[0]

    sample=pd.read_sql_query(f'SELECT * FROM synapse_model LIMIT {int(args.sample)}',con)
    sample.to_csv(out/"synapse_model_sample.csv",index=False,encoding="utf-8-sig")

    # Probe whether pair_id exists; if yes, quantify overlap with primary transferred cells.
    names=set(col_df["name"].astype(str))
    report={
        "status":"COMPLETE",
        "db":str(db),
        "synapse_model_rows":int(n_model),
        "columns":col_df.to_dict(orient="records"),
        "foreign_keys":fk_df.to_dict(orient="records"),
        "pair_id_present":"pair_id" in names,
    }

    if "pair_id" in names:
        # Join only pair/cell/slice metadata; never touch pulse_response.
        q = """
        SELECT
            sm.*,
            p.pre_cell_id,
            p.post_cell_id,
            p.has_synapse,
            p.distance,
            p.experiment_id
        FROM synapse_model sm
        LEFT JOIN pair p ON p.id = sm.pair_id
        """
        model_pairs=pd.read_sql_query(q,con)

        comp=pd.read_csv(transfer,low_memory=False)
        comp=comp[["cell_id","C_transfer","transfer_domain_primary","species"]].copy()
        pre=comp.rename(columns={
            "cell_id":"pre_cell_id",
            "species":"transfer_species"
        })
        model_pairs=model_pairs.merge(pre,on="pre_cell_id",how="left")

        primary=model_pairs[
            (model_pairs["transfer_species"].astype(str).str.lower()=="mouse") &
            model_pairs["transfer_domain_primary"].fillna(False)
        ].copy()

        model_pairs.to_csv(out/"synapse_model_pair_join.csv",index=False,encoding="utf-8-sig")
        primary.to_csv(out/"synapse_model_mouse_transfer_overlap.csv",index=False,encoding="utf-8-sig")

        report["pair_join_rows"]=int(len(model_pairs))
        report["unique_pairs"]=int(model_pairs["pair_id"].nunique())
        report["mouse_transfer_overlap_rows"]=int(len(primary))
        report["mouse_transfer_overlap_pairs"]=int(primary["pair_id"].nunique())
        report["mouse_transfer_overlap_pre_cells"]=int(primary["pre_cell_id"].nunique())

        # Numeric completeness/range inventory for deciding which model parameters
        # can support standardized train simulation.
        numeric=[]
        for c in primary.columns:
            x=pd.to_numeric(primary[c],errors="coerce")
            nn=int(x.notna().sum())
            if nn==0: continue
            numeric.append({
                "column":c,
                "n_nonnull":nn,
                "fraction_nonnull":float(nn/max(len(primary),1)),
                "n_unique":int(x.nunique(dropna=True)),
                "min":float(x.min()),
                "median":float(x.median()),
                "max":float(x.max()),
            })
        pd.DataFrame(numeric).sort_values(
            ["fraction_nonnull","n_unique"],ascending=False
        ).to_csv(out/"synapse_model_numeric_qc.csv",index=False,encoding="utf-8-sig")

    (out/"audit_report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"
    )

    lines=[
        "# Stage 5C synapse-model targeted audit v1","",
        f"- `synapse_model` rows: **{n_model:,}**",
        f"- columns: **{len(col_df)}**",
        f"- `pair_id` present: **{'pair_id' in names}**","",
        "## Scientific purpose","",
        "This audit is the gate before defining standardized train-based dynamic leverage. "
        "No new complexity score is fitted and no raw pulse-response table is scanned.","",
        "## Next decision","",
        "If the model table exposes identifiable release/depression/facilitation/kinetic "
        "parameters with sufficient overlap to the frozen mouse transfer cohort, the next "
        "stage will simulate the same presynaptic spike train for every synapse and compute "
        "`E_ij^train`, followed by presynaptic local leverage analyses."
    ]
    if "mouse_transfer_overlap_pairs" in report:
        lines.insert(5,f"- mouse frozen-transfer overlap pairs: **{report['mouse_transfer_overlap_pairs']:,}**")
        lines.insert(6,f"- mouse frozen-transfer presynaptic cells: **{report['mouse_transfer_overlap_pre_cells']:,}**")

    (out/"00_AUDIT_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    con.close()

    print("="*96)
    print("Stage 5C synapse-model targeted audit v1 COMPLETE")
    print("synapse_model rows:",n_model)
    print("columns:",len(col_df))
    if "mouse_transfer_overlap_pairs" in report:
        print("mouse transfer overlap pairs:",report["mouse_transfer_overlap_pairs"])
        print("mouse transfer pre cells:",report["mouse_transfer_overlap_pre_cells"])
    print("Output:",out)
    print("="*96)

if __name__=="__main__":
    main()
