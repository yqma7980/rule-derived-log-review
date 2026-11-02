"""Input-only screening of official KGS 2001-2005 received-LAS archive.

No rule outputs or model predictions are used in file selection.
"""
from pathlib import Path
import argparse, collections, hashlib, io, json, re, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = "https://www.kgs.ku.edu/PRS/Scans/Log_Summary/2001_2005.zip"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--inventory', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if not args.archive.is_file() or not args.inventory.is_file():
    parser.error('Archive and scanner inventory must both exist')
if hashlib.sha256(args.archive.read_bytes()).hexdigest() != '325059152a76c89fb1a798390fe955f6256e07e00e9a40305691122bcc940edd':
    parser.error('Archive SHA-256 differs from the frozen input')
ROOT = args.output_dir.resolve()
for name in ('kgs_selected_wells.csv', 'input_manifest.json', 'input_summary.json'):
    if (ROOT / name).exists():
        parser.error('Prepared outputs exist; choose a fresh output directory')
inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
if not isinstance(inventory, list) or not inventory:
    parser.error('Inventory must be a nonempty scanner record list')
ROOT.mkdir(parents=True, exist_ok=True)
manifest, frames = [], {}
with zipfile.ZipFile(args.archive) as outer:
    for rec in inventory:
        row = {"archive_member": rec["archive"], "las_file": rec["file"],
               "well_name": rec["well"].get("WELL", ""),
               "api_raw": rec["well"].get("API", ""),
               "well": re.sub(r"\D", "", rec["well"].get("API", ""))[:10],
               "source_url": SOURCE, "header": rec["well"]}
        with zipfile.ZipFile(io.BytesIO(outer.read(rec["archive"]))) as inner:
            raw = inner.read(rec["file"])
        row["las_sha256"] = hashlib.sha256(raw).hexdigest()
        txt = raw.decode("latin1")
        curves = rec["curves"]
        names = [x[0] for x in curves]
        gi = [i for i,n in enumerate(names) if n == "GR"]
        di = [i for i,n in enumerate(names) if n == "RHOB"]
        zi = [i for i,n in enumerate(names) if n in ("DEPT", "DEPTH")]
        row["curves"] = curves
        if len(gi)!=1 or len(di)!=1 or len(zi)!=1:
            row["status"]="excluded_missing_or_ambiguous_GR_RHOB_DEPTH"; manifest.append(row);continue
        gi,di,zi=gi[0],di[0],zi[0]
        row.update(gr_unit=curves[gi][1], den_unit=curves[di][1], depth_unit=curves[zi][1])
        if curves[gi][1].upper() not in ("API","GAPI") or curves[di][1].upper() not in ("G/CC","GM/CC","G/C3","GM/C3","GCC") or curves[zi][1].upper() not in ("F","FT"):
            row["status"]="excluded_missing_or_unrecognized_input_units";manifest.append(row);continue
        if len(row["well"])!=10:
            row["status"]="excluded_missing_API";manifest.append(row);continue
        lines = txt.splitlines()
        start = next(i for i,l in enumerate(lines) if l.strip().upper().startswith("~A"))+1
        data = [l for l in lines[start:] if l.strip() and not l.lstrip().startswith(("#","~"))]
        # Fixed n-curve reshape supports both ordinary rows and wrapped LAS data.
        vals = np.fromstring("\n".join(data), sep=" ")
        if vals.size%len(curves):
            row["status"]="excluded_numeric_column_count";row["numeric_count"]=int(vals.size);manifest.append(row);continue
        arr=vals.reshape(-1,len(curves))
        null=float(rec["well"].get("NULL","-999.25"))
        z,g,d=arr[:,zi]*0.3048,arr[:,gi].copy(),arr[:,di].copy()
        g[np.isclose(g,null)]=np.nan;d[np.isclose(d,null)]=np.nan
        z[np.isclose(arr[:,zi],null)]=np.nan
        # Preserve every row and gap; only count valid rows for duplicate-file choice.
        valid=np.isfinite(z)&np.isfinite(g)&np.isfinite(d)&(g>0)&(d>0)&(d<=5)
        zd=np.sort(np.unique(z[np.isfinite(z)]))
        steps=np.diff(zd);step=float(np.median(steps[steps>0])) if len(steps) else None
        row.update(status="candidate",n_rows=int(len(z)),n_joint_valid=int(valid.sum()),
                   depth_min_m=float(np.nanmin(z)),depth_max_m=float(np.nanmax(z)),
                   median_unique_step_m=step,duplicate_depth_rows=int(len(z[np.isfinite(z)])-len(zd)))
        frames[rec["file"]]=pd.DataFrame({"well":row["well"],"depth":z,"gr":g,"den":d})
        manifest.append(row)

grouped=collections.defaultdict(list)
for rec in manifest:
    if rec["status"]=="candidate":grouped[rec["well"]].append(rec)
selected=[]
for well,recs in sorted(grouped.items()):
    order=sorted(recs,key=lambda r:(-r["n_joint_valid"],r["las_file"]))
    order[0]["status"]="selected";selected.append(order[0])
    for other in order[1:]:
        other["status"]="excluded_duplicate_API_fewer_joint_valid_rows"
        other["selected_file_for_API"]=order[0]["las_file"]
if not selected:
    parser.error('No LAS inputs meet the documented units and column requirements')
out=pd.concat([frames[r["las_file"]] for r in selected],ignore_index=True)
out.to_csv(ROOT/"kgs_selected_wells.csv",index=False)
(ROOT/"input_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
summary={"source_url":SOURCE,"archive_bytes":args.archive.stat().st_size,
 "archive_sha256":hashlib.sha256(args.archive.read_bytes()).hexdigest(),
 "date_window_meaning":"2001-2005 are KGS received/archive years, not necessarily logging years",
 "curve_inventory_candidates":len(inventory),"selected_wells":len(selected),"selected_rows":len(out),
 "joint_valid_rows_before_segment_QC":sum(r["n_joint_valid"] for r in selected),
 "status_counts":dict(collections.Counter(r["status"] for r in manifest)),
 "step_counts_m":dict(collections.Counter(str(round(r["median_unique_step_m"],6)) for r in selected)),
 "selection":"Explicit GR API/GAPI and RHOB bulk-density g/cm3 units; depth ft converted by 0.3048. Same API: most jointly valid input rows, tie by LAS filename. No rule/model output examined.",
 "csv_sha256":hashlib.sha256((ROOT/"kgs_selected_wells.csv").read_bytes()).hexdigest()}
(ROOT/"input_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
