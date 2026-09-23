"""
log_parser.py 

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr

Parser for hwpctl.py log files.

Extracts, for every external GOTO command sent to the HWP controller

Parsing rules:

- A command block starts at a line containing "[ CMD: GOTO <target> ]" and
  ends at whichever comes first: the matching "SUCCESS going to position
  <target>" line, or the next "HWP CONTROL SOFTWARE" restart banner.
- Total steps is the sum of every "moving N steps" / "moved N steps"
  occurrence inside the block.
- A direction change is counted whenever a "set_direction()" line does not
  carry the "(already in spin direction - return with no delay)" suffix.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from astropy.table import Table, MaskedColumn

# --- Regex patterns used to identify relevant log content -----------------

TIMESTAMP_RE = re.compile(r"^(\d{6}):\s?(.*)$")
CMD_RE = re.compile(r"\[\s*CMD:\s*GOTO\s*(\d+)\s*\]")
SUCCESS_RE = re.compile(r"SUCCESS going to position (\d+)")
ALREADY_AT_POS_RE = re.compile(r"already pos \((\d+)\)\s*==\s*pos_target \((\d+)\)")
START_READING_POS_RE = re.compile(r"start reading pos (-?\d+)")
RESTART_RE = re.compile(r"HWP CONTROL SOFTWARE")
STEPS_RE = re.compile(r"(?:moving|moved)\s+(\d+)(?:\s+\w+)?\s+steps", re.IGNORECASE)
SET_DIRECTION_RE = re.compile(r"set_direction\(\):")
ALREADY_IN_SPIN_RE = re.compile(r"already in spin direction", re.IGNORECASE)
POS_TO_ZERO_RE = re.compile(r"pos_to_zero\(\)-SUCCESS: found 0 after moving (\d+) steps, then moved (\d+) ensure_zero steps")
ZERO_TO_EDGE_RE = re.compile(r"zero_to_edge\(\)-SUCCESS: moved (\d+) steps reading zero in the allowed range")
DIRECTION_VALUE_RE = re.compile(r"set_direction\(\):\s*(?:First direction set\s+)?(\d+-->\d+)")
FIRST_VALID_POS_RE = re.compile(r"first valid position read is pos (-?\d+)")
POS_MATCH_TARGET_RE = re.compile(r"pos \((\d+)\)\s*==\s*pos_target \((\d+)\)")
ENGAGE_FAILED_RE = re.compile(r"HWP ENGAGE FAILED!")
SECOND_ATTEMPT_FAILED_RE = re.compile(r"ERROR in the second attempt to set position (\d+)")

OUTPUT_COLUMNS = [
    "command_index",
    "start_line",
    "end_line",
    "start_time",
    "end_time",
    "start_position",
    "target_position",
    "outcome",
    "failure_reason",
    "n_steps",
    "pos_to_zero_steps",
    "ensure_zero_steps",
    "zero_to_edge_steps",
    "direction_changed",
    "n_direction_changes",
    "recovered_after_restart",
    "position_before_restart",
    "direction_before_restart",
    "direction_after_restart",
    "first_position_after_engage",
]

# Columns holding integer counts/indices/positions rather than floats or
# strings. These get an explicit int cast plus a mask for missing values,
# since a raw None alongside ints would otherwise force a float or object
# column when building the table.
INT_COLUMNS = [
    "command_index",
    "start_line",
    "end_line",
    "start_position",
    "target_position",
    "n_steps",
    "pos_to_zero_steps",
    "ensure_zero_steps",
    "zero_to_edge_steps",
    "n_direction_changes",
    "position_before_restart",
    "direction_before_restart",
    "direction_after_restart",
    "first_position_after_engage",
]


def _timestamp_to_seconds(timestamp_str, day_offset_s):
    """Convert an HHMMSS timestamp string to seconds-of-day plus day offset."""
    hours, minutes, seconds = int(timestamp_str[0:2]), int(timestamp_str[2:4]), int(timestamp_str[4:6])
    return day_offset_s + hours * 3600 + minutes * 60 + seconds


def _load_lines(log_path):
    """
    Split a log file into (timestamp_str, time_s, content, line_no) tuples.

    Lines without a leading HHMMSS timestamp (e.g. wrapped continuation
    lines from a multi-line log call) are kept with time_s=None so they
    can still be scanned for content but never drive time calculations.
    """
    parsed_lines = []
    day_offset_s = 0.0
    previous_time_s = None

    for line_no, raw_line in enumerate(log_path.read_text().splitlines(), start=1):
        match = TIMESTAMP_RE.match(raw_line)
        if not match:
            parsed_lines.append((None, None, raw_line, line_no))
            continue

        timestamp_str, content = match.groups()
        time_s = _timestamp_to_seconds(timestamp_str, day_offset_s)

        # Handle midnight rollover so later time differences stay positive.
        if previous_time_s is not None and time_s < previous_time_s:
            day_offset_s += 86400
            time_s += 86400
        previous_time_s = time_s

        parsed_lines.append((timestamp_str, time_s, content, line_no))

    return parsed_lines


def _new_command(command_index, target_position, line_no, timestamp_str, time_s, recovered_after_restart):
    """Initialize the bookkeeping dict for a newly opened command block."""
    return {
        "command_index": command_index,
        "start_line": line_no,
        "end_line": None,
        "start_time": timestamp_str,
        "end_time": None,
        "start_position": None,
        "target_position": target_position,
        "outcome": None,
        "failure_reason": None,
        "n_steps": 0,
        "pos_to_zero_steps": None,
        "ensure_zero_steps": None,
        "zero_to_edge_steps": None,
        "n_direction_changes": 0,
        "recovered_after_restart": recovered_after_restart,
        "position_before_restart": None,
        "direction_before_restart": None,
        "direction_after_restart": None,
        "first_position_after_engage": None,
        "last_seen_time_s": time_s,
        "_n_pos_to_zero_events": 0,
        "_n_zero_to_edge_events": 0,
    }


def _finalize(open_cmd):
    """Add the derived boolean field and drop internal bookkeeping keys."""
    record = dict(open_cmd)
    record["direction_changed"] = record["n_direction_changes"] > 0
    record.pop("last_seen_time_s", None)
    record.pop("_n_pos_to_zero_events", None)
    record.pop("_n_zero_to_edge_events", None)
    return record


def _extract_confirmed_position(text):
    """
    Return the HWP position confirmed as successfully reached in this log
    line, or None if the line reports no such confirmation.

    Only two phrasings unambiguously confirm a successful arrival: the
    final "SUCCESS going to position N" line, and the "pos (N) ==
    pos_target (N)" match reported by set_position() (covering both the
    case where the target was reached after an actual movement and the
    case where it was already the current position). Intermediate or
    stalled readings (position 0, meaning the HWP is spinning or stopped
    between positions, or any failed search) are not confirmations and
    are ignored; only 1-7 are valid physical positions.
    """
    success_match = SUCCESS_RE.search(text)
    if success_match:
        position = int(success_match.group(1))
        return position if 1 <= position <= 7 else None

    pos_match = POS_MATCH_TARGET_RE.search(text)
    if pos_match:
        position = int(pos_match.group(1))
        return position if 1 <= position <= 7 else None

    return None


def _update_last_known_direction(text, last_direction):
    """Return the most recently commanded spin direction string, or last_direction unchanged."""
    direction_match = DIRECTION_VALUE_RE.search(text)
    if direction_match:
        return direction_match.group(1)
    return last_direction


def _direction_string_to_sign(direction_str):
    """
    Convert a raw 'A-->B' direction string into a signed step: +1 when
    moving towards higher position numbers (e.g. 1-->7), -1 when moving
    towards lower ones (e.g. 7-->1). Returns None if direction_str is None.
    """
    if direction_str == "1-->7":
        return 1
    if direction_str == "7-->1":
        return -1
    return None


def parse_hwp_log(log_path):
    """
    Parse an hwpctl.py log file into one record per external GOTO command.

    Parameters
    ----------
    log_path : pathlib.Path
        Path to the log file to analyze.

    Returns
    -------
    astropy.table.Table
        One row per GOTO command. Columns: command_index, start_line,
        end_line, start_time, end_time, start_position, target_position,
        outcome ("success", "failure", or "incomplete"), n_steps,
        pos_to_zero_steps, ensure_zero_steps, zero_to_edge_steps,
        direction_changed, n_direction_changes,
        recovered_after_restart, position_before_restart,
        direction_before_restart, direction_after_restart,
        first_position_after_engage. Integer columns with missing values
        are masked.

        position_before_restart / direction_before_restart are only set
        for commands ended by a server restart ("failure" outcome): the
        last position and spin direction reported anywhere in the log
        before the restart, regardless of which routine reported them.
        direction_after_restart / first_position_after_engage are only
        set for the command right after a restart
        (recovered_after_restart=True): the first commanded direction and
        the first position engage_routine() manages to read back once the
        controller re-engages.
    """
    lines = _load_lines(log_path)

    commands = []
    command_index = 0
    open_cmd = None
    just_restarted = False
    last_reached_position = None
    last_known_direction = None

    for timestamp_str, time_s, text, line_no in lines:
        if text is None:
            continue

        if RESTART_RE.search(text):
            if open_cmd is not None:
                # Server died mid-command: this GOTO attempt is a failure.
                open_cmd["outcome"] = "failure"
                open_cmd["failure_reason"] = "server_restart"
                open_cmd["end_line"] = line_no
                open_cmd["end_time"] = timestamp_str
                open_cmd["position_before_restart"] = last_reached_position
                open_cmd["direction_before_restart"] = _direction_string_to_sign(last_known_direction)
                commands.append(_finalize(open_cmd))
                open_cmd = None
            just_restarted = True
            continue
        
        if open_cmd is not None and ENGAGE_FAILED_RE.search(text):
            open_cmd["outcome"] = "failure"
            open_cmd["failure_reason"] = "engage_failed"
            open_cmd["end_line"] = line_no
            open_cmd["end_time"] = timestamp_str
            commands.append(_finalize(open_cmd))
            open_cmd = None
            continue

        if open_cmd is not None and SECOND_ATTEMPT_FAILED_RE.search(text):
            open_cmd["outcome"] = "failure"
            open_cmd["failure_reason"] = "second_attempt_failed"
            open_cmd["end_line"] = line_no
            open_cmd["end_time"] = timestamp_str
            commands.append(_finalize(open_cmd))
            open_cmd = None
            continue

        cmd_match = CMD_RE.search(text)
        if cmd_match:
            if open_cmd is not None:
                # A new command appeared before the previous one was resolved.
                # Close the previous one as "incomplete" rather than silently
                # dropping it, since this should not happen in a well-formed log.
                open_cmd["outcome"] = "incomplete"
                open_cmd["end_line"] = line_no - 1
                commands.append(_finalize(open_cmd))

            command_index += 1
            open_cmd = _new_command(
                command_index, int(cmd_match.group(1)), line_no, timestamp_str, time_s, just_restarted
            )
            just_restarted = False
            continue

        # Position and direction are tracked continuously, independent of
        # command boundaries, since the last successful arrival or commanded
        # direction remains valid state until a newer confirmation replaces it.
        confirmed_position = _extract_confirmed_position(text)
        if confirmed_position is not None:
            last_reached_position = confirmed_position
        last_known_direction = _update_last_known_direction(text, last_known_direction)

        if open_cmd is None:
            # Lines outside any tracked command (restart banner internals,
            # stray output, etc.) carry no command to attach to.
            continue

        if time_s is not None:
            open_cmd["last_seen_time_s"] = time_s

        # Starting position: first "start reading pos N" match, falling back
        # to the "already pos (N) == pos_target (N)" no-op case.
        if open_cmd["start_position"] is None:
            start_match = START_READING_POS_RE.search(text)
            if start_match:
                open_cmd["start_position"] = int(start_match.group(1))
            else:
                already_match = ALREADY_AT_POS_RE.search(text)
                if already_match:
                    open_cmd["start_position"] = int(already_match.group(1))

        # For the command right after a restart, capture the first commanded
        # direction and the first position engage_routine() reads back, to
        # help diagnose whether the previous failure overshot or undershot
        # its target.
        if open_cmd["recovered_after_restart"]:
            if open_cmd["direction_after_restart"] is None:
                direction_match = DIRECTION_VALUE_RE.search(text)
                if direction_match:
                    open_cmd["direction_after_restart"] = _direction_string_to_sign(direction_match.group(1))
            if open_cmd["first_position_after_engage"] is None:
                first_valid_match = FIRST_VALID_POS_RE.search(text)
                if first_valid_match:
                    open_cmd["first_position_after_engage"] = int(first_valid_match.group(1))

        for steps_match in STEPS_RE.finditer(text):
            open_cmd["n_steps"] += int(steps_match.group(1))

        # pos_to_zero and ensure_zero steps are reported together on the same
        # line. Only keep them if this is the first such event in the block;
        # a second occurrence means the position was not found on the first
        # attempt, so the per-stage breakdown becomes ambiguous and is left
        # missing (the total in n_steps still accounts for every attempt).
        pos_to_zero_match = POS_TO_ZERO_RE.search(text)
        if pos_to_zero_match:
            open_cmd["_n_pos_to_zero_events"] += 1
            if open_cmd["_n_pos_to_zero_events"] == 1:
                open_cmd["pos_to_zero_steps"] = int(pos_to_zero_match.group(1))
                open_cmd["ensure_zero_steps"] = int(pos_to_zero_match.group(2))
            else:
                open_cmd["pos_to_zero_steps"] = None
                open_cmd["ensure_zero_steps"] = None

        zero_to_edge_match = ZERO_TO_EDGE_RE.search(text)
        if zero_to_edge_match:
            open_cmd["_n_zero_to_edge_events"] += 1
            if open_cmd["_n_zero_to_edge_events"] == 1:
                open_cmd["zero_to_edge_steps"] = int(zero_to_edge_match.group(1))
            else:
                open_cmd["zero_to_edge_steps"] = None

        if SET_DIRECTION_RE.search(text) and not ALREADY_IN_SPIN_RE.search(text):
            open_cmd["n_direction_changes"] += 1

        success_match = SUCCESS_RE.search(text)
        if success_match:
            open_cmd["outcome"] = "success"
            open_cmd["end_line"] = line_no
            open_cmd["end_time"] = timestamp_str
            commands.append(_finalize(open_cmd))
            open_cmd = None

    if open_cmd is not None:
        # File ends mid-command (truncated log): keep the partial record
        # rather than discarding data already collected for it.
        open_cmd["outcome"] = "incomplete"
        open_cmd["end_line"] = lines[-1][3]
        commands.append(_finalize(open_cmd))

    return _records_to_table(commands)


def _records_to_table(commands):
    """
    Convert the list of per-command dicts into an astropy Table.

    Integer columns are built as MaskedColumn with dtype int and an
    explicit mask for missing values (None), since a raw list containing
    None alongside ints would otherwise force a float or object column.
    """
    table = Table()

    for col_name in OUTPUT_COLUMNS:
        column_values = [record[col_name] for record in commands]

        if col_name in INT_COLUMNS:
            mask = [value is None for value in column_values]
            filled_values = [0 if value is None else int(value) for value in column_values]
            table[col_name] = MaskedColumn(data=np.array(filled_values, dtype=int), mask=mask)
        else:
            table[col_name] = column_values

    return table


def print_summary(commands_table):
    """Print a quick sanity-check summary of the parsed commands to stdout."""
    n_total = len(commands_table)
    outcomes = commands_table["outcome"]

    print(f"Parsed {n_total} GOTO commands")
    for outcome in np.unique(outcomes):
        count = int(np.sum(outcomes == outcome))
        print(f"  {outcome}: {count}")

    resolved_mask = np.isin(outcomes, ["success", "failure"])
    resolved = commands_table[resolved_mask]
    if len(resolved) > 0:
        success_rate = np.mean(resolved["outcome"] == "success") * 100
        print(f"Success rate (excluding incomplete): {success_rate:.1f}%")


    n_direction_changes = int(np.sum(commands_table["direction_changed"]))
    print(f"Commands with at least one direction reversal: {n_direction_changes} / {n_total}")


def main():
    """Command-line entry point: parse a log file and save results to CSV."""
    parser = argparse.ArgumentParser(description="Parse hwpctl.py log files into per-command records.")
    parser.add_argument("log_path", type=Path, help="Path to the hwpctl.py log file to analyze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output CSV file. Defaults to <log_path stem>_parsed.csv in the same directory.",
    )
    args = parser.parse_args()

    output_path = args.output if args.output is not None else args.log_path.with_name(args.log_path.stem + "_parsed.csv")

    commands_table = parse_hwp_log(args.log_path)
    commands_table.write(output_path, format="ascii.csv", overwrite=True)

    print_summary(commands_table)
    print(f"\nSaved parsed commands to {output_path}")


if __name__ == "__main__":
    main()
