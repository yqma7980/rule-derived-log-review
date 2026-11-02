"""Scan the official KGS 2001-2005 archive before measurement-only screening.

This recovers gr_density_inventory.json for prepare_kgs.py. It never computes
rule outputs, change labels, model predictions, or performance measures.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

EXPECTED_SHA256 = "325059152a76c89fb1a798390fe955f6256e07e00e9a40305691122bcc940edd"
SOURCE_URL = "https://www.kgs.ku.edu/PRS/Scans/Log_Summary/2001_2005.zip"


def scan(archive, output):
    actual_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError("Archive SHA-256 differs from the frozen input. Do not overwrite an existing study with a changed download.")
    if any((output / name).exists() for name in ('gr_density_inventory.json', 'scan_summary.json')):
        raise FileExistsError('Scanner outputs exist; choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    curves_count = collections.Counter()
    las_count = 0
    with zipfile.ZipFile(archive) as outer:
        archive_members = len(outer.infolist())
        for member in outer.infolist():
            if member.is_dir():
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                for filename in inner.namelist():
                    if not filename.lower().endswith(".las"):
                        continue
                    las_count += 1
                    text = inner.read(filename).decode("latin1")
                    lines = text.splitlines()
                    section = ""
                    curves, header = [], {}
                    data_start = None
                    for line_number, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith("~"):
                            section = stripped[1:2].upper()
                        elif stripped and not stripped.startswith("#"):
                            if section == "C":
                                match = re.match(r"\s*([^\.]+)\.\s*([^\s:]*)", line)
                                if match:
                                    curves.append((match[1].strip().upper(), match[2]))
                            elif section == "W":
                                match = re.match(r"\s*([^\.]+)\.([^:]*):?(.*)", line)
                                if match:
                                    header[match[1].strip().upper()] = match[2].strip()
                            elif section == "A":
                                data_start = line_number
                                break
                    curves_count.update(curve[0] for curve in curves)
                    gamma = [c for c in curves if c[0] in ("GR", "GAM", "SGR")]
                    density = [c for c in curves if c[0] in ("RHOB", "RHOZ", "DEN", "DENS", "RHO")]
                    if gamma and density:
                        if data_start is None:
                            raise ValueError(f"Candidate {filename} has no ASCII data section")
                        records.append({"archive": member.filename, "file": filename,
                                        "curves": curves, "well": header,
                                        "rows": len(lines) - data_start})
    (output / "gr_density_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"source_url": SOURCE_URL, "archive_sha256": actual_sha256,
               "archive_bytes": archive.stat().st_size, "archive_members": archive_members,
               "las_files_scanned": las_count, "candidate_las_files": len(records),
               "candidate_unique_raw_API": len({r["well"].get("API", "") for r in records}),
               "curve_counts": dict(curves_count),
               "screening_note": "Alias-based header inventory only. prepare_kgs.py subsequently requires unambiguous GR/RHOB/DEPTH with explicit units and chooses one record per API using input availability only."}
    (output / "scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent / "2001_2005.zip")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = scan(args.archive, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "curve_counts"}, indent=2))
