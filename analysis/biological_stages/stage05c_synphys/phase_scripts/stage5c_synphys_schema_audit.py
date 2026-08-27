#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, json, os, re, sqlite3, time
from pathlib import Path
from urllib.parse import quote

DEFAULT_DB = Path.cwd() / "synphys_r2.1_full.sqlite"

CATEGORY_PATTERNS = {
    "cell_identity": [r"\bcell\b", r"cre", r"transgenic", r"layer", r"species", r"morph", r"cortical", r"location", r"dendrite"],
    "intrinsic_physiology": [r"intrinsic", r"electrophys", r"ephys", r"spike", r"threshold", r"rheobase", r"adapt", r"input_res", r"capacit", r"\btau\b", r"sag", r"resting", r"firing", r"trough", r"width", r"amplitude"],
    "experiment_slice": [r"experiment", r"slice", r"recording", r"rig", r"acsf", r"internal", r"solution", r"age", r"temperature"],
    "pair_connection": [r"\bpair\b", r"pre_", r"post_", r"connection", r"connected", r"distance", r"gap", r"electrical"],
    "synapse_strength": [r"synapse", r"psp", r"psc", r"amplitude", r"latency", r"rise", r"decay", r"conductance", r"resting.*state", r"fit"],
    "dynamics_stp": [r"dynamic", r"stp", r"plastic", r"facil", r"depress", r"paired.*pulse", r"train", r"recovery"],
    "pulse_response": [r"pulse", r"response", r"stim", r"spike.*train", r"baseline"],
}

STAGE5A_TERMS = [
    "threshold","trough","fast_trough","adaptation","f_i","fi","spike_width","width",
    "amplitude","rheobase","input_resistance","input_res","tau","sag","resting","firing_rate","spike"
]

def qident(name):
    return '"' + name.replace('"','""') + '"'

def ro_connect(path):
    uri = f"file:{quote(str(path).replace(os.sep,'/'))}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con

def classify(text):
    t = text.lower()
    out = {}
    for cat, pats in CATEGORY_PATTERNS.items():
        hits=[p for p in pats if re.search(p,t,re.I)]
        if hits: out[cat]=hits
    return out

def overlap(name):
    n=name.lower()
    return [x for x in STAGE5A_TERMS if x in n]

def write_csv(path, rows):
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]
    seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def row_estimates(con):
    out={}
    try:
        for tbl,stat in con.execute("SELECT tbl, stat FROM sqlite_stat1"):
            if stat:
                try: out[tbl]=max(out.get(tbl,0),int(str(stat).split()[0]))
                except: pass
    except:
        pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=DEFAULT_DB)
    ap.add_argument("--out",type=Path,default=None)
    ap.add_argument("--sample-rows",type=int,default=5)
    ap.add_argument("--exact-counts",action="store_true")
    ap.add_argument("--exact-count-max",type=int,default=1_000_000)
    args=ap.parse_args()

    db=args.db.resolve()
    if not db.exists():
        raise SystemExit(f"Database not found: {db}")
    out=(args.out or (db.parent/"stage5c_synphys_schema_audit")).resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/"create_sql").mkdir(exist_ok=True)

    print("="*90)
    print("Stage 5C Synaptic Physiology schema audit")
    print("DB:",db)
    print(f"Size: {db.stat().st_size/1024**3:.2f} GiB")
    print("READ-ONLY")
    print("="*90)

    t0=time.time()
    con=ro_connect(db)
    page_size=con.execute("PRAGMA page_size").fetchone()[0]
    page_count=con.execute("PRAGMA page_count").fetchone()[0]

    master=con.execute("""
        SELECT type,name,tbl_name,sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type,name
    """).fetchall()
    tables=[r for r in master if r[0]=="table"]
    views=[r for r in master if r[0]=="view"]
    estimates=row_estimates(con)

    table_rows=[]; column_rows=[]; fk_rows=[]; index_rows=[]
    candidate_tables=[]; candidate_columns=[]; samples={}

    for i,(_,table,_,create_sql) in enumerate(tables,1):
        print(f"[{i}/{len(tables)}] {table}")
        if create_sql:
            safe=re.sub(r"[^A-Za-z0-9_.-]+","_",table)
            (out/"create_sql"/f"{safe}.sql").write_text(create_sql+";\n",encoding="utf-8")

        cols=con.execute(f"PRAGMA table_info({qident(table)})").fetchall()
        fks=con.execute(f"PRAGMA foreign_key_list({qident(table)})").fetchall()
        idxs=con.execute(f"PRAGMA index_list({qident(table)})").fetchall()

        est=estimates.get(table)
        exact=None; count_status="NOT_RUN"
        if args.exact_counts and est is not None and est<=args.exact_count_max:
            try:
                exact=con.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0]
                count_status="EXACT"
            except Exception as e:
                count_status=f"FAILED:{type(e).__name__}"

        text=" ".join([table,create_sql or ""]+[str(c[1]) for c in cols])
        hits=classify(text)
        table_rows.append({
            "table":table,"n_columns":len(cols),"row_estimate_sqlite_stat1":est,
            "exact_count":exact,"count_status":count_status,"n_foreign_keys":len(fks),
            "n_indexes":len(idxs),"candidate_categories":";".join(sorted(hits))
        })
        if hits:
            candidate_tables.append({
                "table":table,"categories":";".join(sorted(hits)),
                "matched_patterns":json.dumps(hits,ensure_ascii=False),
                "row_estimate":est,"n_columns":len(cols)
            })

        for cid,name,ctype,notnull,dflt,pk in cols:
            ch=classify(f"{table} {name} {ctype}")
            ov=overlap(name)
            rec={"table":table,"cid":cid,"column":name,"type":ctype,"notnull":notnull,
                 "default":dflt,"primary_key":pk,
                 "candidate_categories":";".join(sorted(ch)),
                 "stage5a_overlap_terms":";".join(ov)}
            column_rows.append(rec)
            if ch or ov: candidate_columns.append(rec.copy())

        for fk in fks:
            fk_rows.append({"table":table,"fk_id":fk[0],"seq":fk[1],"ref_table":fk[2],
                            "from_column":fk[3],"to_column":fk[4],"on_update":fk[5],
                            "on_delete":fk[6],"match":fk[7]})
        for idx in idxs:
            idx_name=idx[1]
            try:
                info=con.execute(f"PRAGMA index_info({qident(idx_name)})").fetchall()
                icols=";".join(str(x[2]) for x in info)
            except:
                icols=""
            index_rows.append({"table":table,"index":idx_name,"unique":idx[2],
                               "origin":idx[3],"partial":idx[4],"columns":icols})

        if hits and args.sample_rows>0:
            try:
                cur=con.execute(f"SELECT * FROM {qident(table)} LIMIT {int(args.sample_rows)}")
                names=[d[0] for d in cur.description]
                rs=cur.fetchall()
                samples[table]=[
                    {names[j]:("<BLOB>" if isinstance(v,(bytes,bytearray)) else (None if v is None else str(v)[:300]))
                     for j,v in enumerate(row)}
                    for row in rs
                ]
            except Exception as e:
                samples[table]={"ERROR":repr(e)}

    write_csv(out/"tables.csv",table_rows)
    write_csv(out/"columns.csv",column_rows)
    write_csv(out/"foreign_keys.csv",fk_rows)
    write_csv(out/"indexes.csv",index_rows)
    write_csv(out/"candidate_tables.csv",candidate_tables)
    write_csv(out/"candidate_columns.csv",candidate_columns)
    (out/"candidate_samples.json").write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding="utf-8")

    overview={
        "database":str(db),"file_bytes":db.stat().st_size,"file_gib":db.stat().st_size/1024**3,
        "sqlite_version":sqlite3.sqlite_version,"page_size":page_size,"page_count":page_count,
        "logical_db_bytes":page_size*page_count,"n_tables":len(tables),"n_views":len(views),
        "audit_seconds":time.time()-t0
    }
    (out/"database_overview.json").write_text(json.dumps(overview,indent=2),encoding="utf-8")

    cats={}
    for r in candidate_tables:
        for c in r["categories"].split(";"):
            cats.setdefault(c,[]).append(r["table"])

    lines=["# Stage 5C Synaptic Physiology — schema audit","",
           f"- Database: `{db}`",f"- Size: **{overview['file_gib']:.2f} GiB**",
           f"- Tables: **{len(tables)}**",f"- Views: **{len(views)}**",
           "- Mode: **READ-ONLY**","",
           "## Candidate table groups",""]
    for cat in CATEGORY_PATTERNS:
        lines.append(f"### {cat}")
        names=sorted(set(cats.get(cat,[])))
        lines += [f"- `{n}`" for n in names] if names else ["- No automatic match."]
        lines.append("")
    lines += ["## Candidate Stage 5A feature-overlap columns",""]
    sc=[r for r in candidate_columns if r["stage5a_overlap_terms"]]
    lines += [f"- `{r['table']}.{r['column']}` ({r['type']}) ← `{r['stage5a_overlap_terms']}`" for r in sc] if sc else ["- No lexical overlap detected."]
    lines += ["","## Row-count policy","",
              "Default audit avoids blind `COUNT(*)` over the 249 GiB DB. "
              "`sqlite_stat1` estimates are used when available. Use `--exact-counts` only after we identify the candidate tables.",
              "","## Next step","",
              "Use this real schema to build the frozen Stage 5A complexity-transfer matrix and "
              "`L_struct`, `L_weight`, and `L_dyn`. Do not guess field semantics from names alone."]
    (out/"00_AUDIT_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    con.close()
    print("\nAUDIT COMPLETE")
    print("Summary:",out/"00_AUDIT_SUMMARY.md")
    print("Candidate tables:",out/"candidate_tables.csv")
    print("Candidate columns:",out/"candidate_columns.csv")
    print("Samples:",out/"candidate_samples.json")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
