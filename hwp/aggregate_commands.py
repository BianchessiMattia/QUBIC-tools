"""
aggregate_commands.py 

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr

Aggregate multiple per-day hwp_log_parser.py outputs into a single combined
table for cross-day analysis of HWP GOTO commands.

Each input file is expected to follow the "<YYYYMMDD>_parsed.csv" naming
convention produced by hwp_log_parser.py. 
"incomplete" commands (truncated log at end of file, or an
anomalous nested CMD) are excluded from the combined table since they do
not represent a resolved success/failure outcome; only their total count
across all input files is reported.
"""

import argparse
import re
from pathlib import Path

from astropy.io.ascii import convert_numpy
from astropy.table import Table, vstack

DATE_FROM_FILENAME_RE = re.compile(r"(\d{8})_parsed\.csv$")

# Timestamp columns whose leading zeros (HHMMSS) must survive the CSV read;
# astropy's default type guessing would otherwise cast them to int and
# silently drop the leading zeros.
TIME_COLUMNS = ["start_time", "end_time"]

# Boolean columns written as literal "True"/"False" text by hwp_log_parser.py
# (via pandas), which astropy reads back as strings rather than bool.
BOOL_COLUMNS = ["direction_changed", "recovered_after_restart"]


def _extract_date(csv_path):
    """
    Extract the observation date encoded in a parsed-log filename.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to a file following the "<YYYYMMDD>_parsed.csv" naming
        convention produced by hwp_log_parser.py.

    Returns
    -------
    str
        The date in ISO format ("YYYY-MM-DD").
    """
    match = DATE_FROM_FILENAME_RE.search(csv_path.name)
    if not match:
        raise ValueError(f"Could not extract a YYYYMMDD date from filename: {csv_path.name}")
    raw_date = match.group(1)
    return f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"


def load_parsed_csv(csv_path):
    """
    Load a single day's hwp_log_parser.py output, restoring column types
    lost on the CSV round-trip, and tag every row with its source date.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to a "<YYYYMMDD>_parsed.csv" file.

    Returns
    -------
    astropy.table.Table
        The per-command table for this day, with an added "date" column.
    """
    converters = {column: [convert_numpy(str)] for column in TIME_COLUMNS}
    converters["failure_reason"] = str
    table = Table.read(csv_path, format="csv", converters=converters)

    for column in BOOL_COLUMNS:
        table[column] = table[column] == "True"

    table["date"] = _extract_date(csv_path)
    return table


def aggregate_parsed_csvs(csv_paths):
    """
    Combine several daily parsed-log tables into one, excluding "incomplete"
    commands from the returned table but keeping their total count.

    Parameters
    ----------
    csv_paths : list of pathlib.Path
        Paths to the daily "<YYYYMMDD>_parsed.csv" files to combine.

    Returns
    -------
    combined_table : astropy.table.Table
        All success/failure commands across every input file, with a "date"
        column identifying the source day. The per-day "command_index" is
        renamed to "command_index_in_day" since it is only unique within a
        single day, not across the combined table.
    n_incomplete : int
        Total number of "incomplete" commands found across all input files,
        excluded from combined_table.
    """
    daily_tables = [load_parsed_csv(csv_path) for csv_path in csv_paths]
    full_table = vstack(daily_tables, metadata_conflicts="silent")

    is_incomplete = full_table["outcome"] == "incomplete"
    n_incomplete = int(is_incomplete.sum())
    combined_table = full_table[~is_incomplete]
    combined_table.rename_column("command_index", "command_index_in_day")

    return combined_table, n_incomplete


def _resolve_csv_paths(inputs):
    """Expand a mix of file and directory arguments into a flat CSV path list."""
    csv_paths = []
    for input_path in inputs:
        if input_path.is_dir():
            csv_paths.extend(sorted(input_path.glob("*_parsed.csv")))
        else:
            csv_paths.append(input_path)
    return csv_paths


def main():
    """Command-line entry point: aggregate daily parsed CSVs into one file."""
    parser = argparse.ArgumentParser(
        description="Combine daily hwp_log_parser.py CSV outputs into a single table for cross-day analysis."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Parsed CSV files and/or directories containing '<YYYYMMDD>_parsed.csv' files to combine.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the combined output CSV. Defaults to 'hwp_commands_combined.csv' in the current directory.",
    )
    args = parser.parse_args()

    csv_paths = _resolve_csv_paths(args.inputs)
    if not csv_paths:
        raise FileNotFoundError("No '*_parsed.csv' files found among the given inputs.")

    combined_table, n_incomplete = aggregate_parsed_csvs(csv_paths)

    output_path = args.output if args.output is not None else Path.cwd() / "hwp_commands_combined.csv"
    combined_table.write(output_path, format="csv", overwrite=True)

    dates = sorted(set(combined_table["date"]))
    outcome_counts = {
        outcome: int((combined_table["outcome"] == outcome).sum()) for outcome in ["success", "failure"]
    }

    print(f"Combined {len(csv_paths)} daily file(s) covering {len(dates)} date(s): {', '.join(dates)}")
    print(f"Commands included in analysis: {len(combined_table)} (success: {outcome_counts['success']}, failure: {outcome_counts['failure']})")
    print(f"Total incomplete commands excluded from analysis: {n_incomplete}")
    print(f"Saved combined table to {output_path}")


if __name__ == "__main__":
    main()
