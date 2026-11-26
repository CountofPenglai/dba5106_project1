"""Count raw missing-value markers in every column, before preprocessing.

The input file documents -99.99 and -999 as missing. Exact -99 is counted
separately: it is not silently classified as an official missing marker.
"""

from portfolio.paths import PROJECT_ROOT

import argparse
import csv
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path

from portfolio.data.preprocessing import DEFAULT_DROP_ASSETS, MISSING_MARKERS
from portfolio.data.returns_loader import MODEL_SECTION


SECTIONS = (
    "Average Value Weighted Returns -- Daily",
    MODEL_SECTION,
    "Number of Firms in Portfolios",
    "Average Firm Size",
)
MARKERS = {-99.0: "count_minus_99", -99.99: "count_minus_99_99", -999.0: "count_minus_999"}


def audit_file(path, *, start=None, end=None):
    """Return all column counts and section coverage; never clean the input."""
    path = Path(path)
    for value in (start, end):
        if value is not None and date.fromisoformat(value).isoformat() != value:
            raise ValueError("Dates must use YYYY-MM-DD format")
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")
    sections, records = {}, []
    section, columns = None, None
    with path.open(encoding="utf-8-sig", newline="") as source:
        for line_number, line in enumerate(source, 1):
            text = line.strip()
            if text in SECTIONS:
                if text in sections:
                    raise ValueError(f"Duplicate section: {text}")
                section, columns = text, None
                sections[section] = {"rows": 0, "first_date": None, "last_date": None,
                    "rows_with_missing": 0, "rows_with_missing_after_default_exclusions": 0,
                    "used_by_model": section == MODEL_SECTION}
                continue
            if not text:
                if columns is not None:
                    section, columns = None, None
                continue
            if section is None:
                continue
            cells = next(csv.reader([line]))
            if columns is None:
                names = [name.strip() for name in cells[1:]]
                if cells[0].strip() or not names or not all(names) or len(set(names)) != len(names):
                    raise ValueError(f"Invalid asset header at line {line_number}")
                columns = [{"section": section, "column_number": i, "asset": asset,
                    "observations": 0, "count_minus_99": 0, "count_minus_99_99": 0,
                    "count_minus_999": 0, "blank_count": 0, "missing_marker_count": 0,
                    "first_marker_date": None, "last_marker_date": None,
                    "excluded_by_default": asset in DEFAULT_DROP_ASSETS}
                    for i, asset in enumerate(names, 1)]
                records.extend(columns)
                continue
            if len(cells) != len(columns) + 1:
                raise ValueError(f"Wrong column count at line {line_number}")
            day = datetime.strptime(cells[0].strip(), "%Y%m%d").date().isoformat()
            if (start is not None and day < start) or (end is not None and day > end):
                continue
            stats = sections[section]
            stats["rows"] += 1
            stats["first_date"] = min(stats["first_date"] or day, day)
            stats["last_date"] = max(stats["last_date"] or day, day)
            row_missing, kept_missing = False, False
            for record, cell in zip(columns, cells[1:]):
                record["observations"] += 1
                value = float(cell) if cell.strip() else math.nan
                missing = math.isnan(value) or value in MISSING_MARKERS
                if math.isnan(value):
                    record["blank_count"] += 1
                elif not math.isfinite(value):
                    raise ValueError(f"Infinite value at line {line_number}")
                elif value in MARKERS:
                    record[MARKERS[value]] += 1
                    if value != -99.0:
                        record["missing_marker_count"] += 1
                        record["first_marker_date"] = min(record["first_marker_date"] or day, day)
                        record["last_marker_date"] = max(record["last_marker_date"] or day, day)
                row_missing |= missing
                kept_missing |= missing and not record["excluded_by_default"]
            stats["rows_with_missing"] += int(row_missing)
            stats["rows_with_missing_after_default_exclusions"] += int(kept_missing)
    if not any(record["section"] == MODEL_SECTION for record in records):
        raise ValueError(f"Missing or empty section: {MODEL_SECTION}")
    for record in records:
        count = record["observations"]
        record["missing_percent"] = 100 * (record["missing_marker_count"] + record["blank_count"]) / count if count else None
    for heading, stats in sections.items():
        stats["assets"] = sum(record["section"] == heading for record in records)
    return {"source": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "requested_start": start, "requested_end": end,
            "official_missing_markers": [-99.99, -999], "exact_minus_99_is_separate": True,
            "default_excluded_assets": list(DEFAULT_DROP_ASSETS),
            "sections": sections, "columns": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", help="Optional inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Optional inclusive YYYY-MM-DD")
    parser.add_argument("--output", type=Path, help="New output directory for counts.csv and audit.json")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("--output must be a new directory")
    try:
        report = audit_file(args.csv, start=args.start, end=args.end)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for heading, stats in report["sections"].items():
        print(f"\n{heading}: {stats['rows']} rows, {stats['first_date']} to {stats['last_date']}")
        rows = [r for r in report["columns"] if r["section"] == heading]
        print("Totals: " + ", ".join(f"{value:g}={sum(r[field] for r in rows)}" for value, field in MARKERS.items()))
        for row in sorted(rows, key=lambda r: (-r["missing_marker_count"], r["column_number"])):
            if row["missing_marker_count"] or row["count_minus_99"] or row["blank_count"]:
                print(f"  {row['column_number']:3d} {row['asset']:12s} "
                      f"-99={row['count_minus_99']:5d} -99.99={row['count_minus_99_99']:5d} "
                      f"-999={row['count_minus_999']:5d} blank={row['blank_count']:5d} "
                      f"excluded={row['excluded_by_default']}")
        print(f"Rows with missing values after default exclusions: {stats['rows_with_missing_after_default_exclusions']}")
    if args.output is not None:
        args.output.mkdir(parents=True)
        (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (args.output / "counts.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(report["columns"][0]))
            writer.writeheader()
            writer.writerows(report["columns"])
        print(f"\nAll columns, including zero counts: {args.output / 'counts.csv'}")



if __name__ == "__main__":
    main()
