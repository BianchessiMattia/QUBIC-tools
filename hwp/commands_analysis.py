"""
commands_analysis.py 

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr

Statistical analysis of HWP GOTO commands (output of hwp_commands_aggregate.py):
Step-count and backslash statistics used for repeatability are computed from completed ("success") commands only.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.ascii import convert_numpy
from astropy.table import Table
from scipy import stats

from aggregate_commands import BOOL_COLUMNS, TIME_COLUMNS


def load_combined_csv(csv_path):
    """
    Load the combined table produced by hwp_commands_aggregate.py,

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the "hwp_commands_combined.csv" file.

    Returns
    -------
    astropy.table.Table
        The combined per-command table with correct dtypes and the
        added "rotation_time" column.
    """
    converters = {column: [convert_numpy(str)] for column in TIME_COLUMNS}
    table = Table.read(csv_path, format="csv", converters=converters)
    for column in BOOL_COLUMNS:
        table[column] = table[column] == "True"

    start_time = [int(t[0:2])*3600 + int(t[2:4])*60 + int(t[4:6]) for t in table["start_time"]]
    end_time = [int(t[0:2])*3600 + int(t[2:4])*60 + int(t[4:6]) for t in table["end_time"]]
    
    table["rotation_time"] = np.array(end_time) - np.array(start_time)

    return table


def compute_restart_diagnosis(table):
    """
    Diagnose, for each server restart, whether the failed GOTO command overshot its target position or stopped short of it.

    Parameters
    ----------
    table : astropy.table.Table
        The combined per-command returned by load_combined_csv().

    Returns
    -------
    astropy.table.Table
        One row per diagnosed restart event, with columns: date,
        command_index_in_day (of the failed command), target_position,
        direction_before_restart, direction_after_restart,
        first_position_after_engage, direction_changed, failure_mode
        ("overshoot", "undershoot", or "unknown").
    """
    diagnosis_rows = []

    for date in np.unique(table["date"]):
        day_table = table[table["date"] == date]
        day_table.sort("start_line")

        for row_index in range(len(day_table) - 1):
            failed_row = day_table[row_index]
            next_row = day_table[row_index + 1]

            if failed_row["outcome"] != "failure" or not next_row["recovered_after_restart"]:
                continue

            direction_before = failed_row["direction_before_restart"]
            direction_after = next_row["direction_after_restart"]
            first_position = next_row["first_position_after_engage"]
            target = failed_row["target_position"]

            # Skip events where any of the required fields could not be
            # extracted from the log (masked value), rather than guessing.
            required_values = [direction_before, direction_after, first_position, target]
            if any(np.ma.is_masked(value) or value is None for value in required_values):
                continue

            direction_changed = direction_after != direction_before

            if not direction_changed:
                if first_position == target + direction_before:
                    failure_mode = "overshoot"
                elif first_position == target:
                    failure_mode = "undershoot"
                else:
                    failure_mode = "unknown"
            else:
                if first_position == target - 1:
                    failure_mode = "undershoot"
                elif first_position == target:
                    failure_mode = "overshoot"
                else:
                    failure_mode = "unknown"

            diagnosis_rows.append({
                "date": date,
                "command_index_in_day": failed_row["command_index_in_day"],
                "target_position": target,
                "direction_before_restart": direction_before,
                "direction_after_restart": direction_after,
                "first_position_after_engage": first_position,
                "direction_changed": direction_changed,
                "failure_mode": failure_mode,
            })

    return Table(rows=diagnosis_rows) if diagnosis_rows else Table(
        names=[
            "date", "command_index_in_day", "target_position", "direction_before_restart",
            "direction_after_restart", "first_position_after_engage",
            "direction_changed", "failure_mode",
        ]
    )


def summarize_restart_diagnosis(restart_diagnosis):
    """
    Summarize failure_mode counts from compute_restart_diagnosis(), overall
    and split by whether the HWP reversed direction after the restart.

    Parameters
    ----------
    restart_diagnosis : astropy.table.Table
        Output of compute_restart_diagnosis().

    Returns
    -------
    astropy.table.Table
        One row per group ("all", "direction_changed=False",
        "direction_changed=True"), with columns: group, n_total,
        n_overshoot, pct_overshoot, n_undershoot, pct_undershoot,
        n_unknown, pct_unknown.
    """
    def _count_row(group_label, subtable):
        n_total = len(subtable)
        n_overshoot = int(np.sum(subtable["failure_mode"] == "overshoot"))
        n_undershoot = int(np.sum(subtable["failure_mode"] == "undershoot"))
        n_unknown = int(np.sum(subtable["failure_mode"] == "unknown"))
        return {
            "group": group_label,
            "n_total": n_total,
            "n_overshoot": n_overshoot,
            "pct_overshoot": 100 * n_overshoot / n_total if n_total > 0 else np.nan,
            "n_undershoot": n_undershoot,
            "pct_undershoot": 100 * n_undershoot / n_total if n_total > 0 else np.nan,
            "n_unknown": n_unknown,
            "pct_unknown": 100 * n_unknown / n_total if n_total > 0 else np.nan,
        }

    summary_rows = [_count_row("all", restart_diagnosis)]
    for direction_changed_value in [False, True]:
        group_label = f"direction_changed={direction_changed_value}"
        subtable = restart_diagnosis[restart_diagnosis["direction_changed"] == direction_changed_value]
        summary_rows.append(_count_row(group_label, subtable))

    return Table(rows=summary_rows)


def compute_position_pair_summary(table):
    """
    Summarize outcome, step-count repeatability, and backslash occurrence
    for every (start_position, target_position) pair present in the table.

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).

    Returns
    -------
    astropy.table.Table
        One row per position pair, sorted by start_position then
        target_position. Columns: start_position, target_position,
        n_total, n_success, n_failure, success_rate_pct,
        mean_n_steps_success, std_n_steps_success, cv_n_steps_success,
        n_direction_changed, direction_change_rate_pct.
    """
    grouped = table.group_by(["start_position", "target_position"])

    start_positions, target_positions = [], []
    n_totals, n_successes, n_failures, success_rates = [], [], [], []
    mean_steps, std_steps, cv_steps = [], [], []
    n_dir_changed, dir_change_rates = [], []

    for group in grouped.groups:
        n_total = len(group)
        is_success = group["outcome"] == "success"
        n_success = int(is_success.sum())

        success_steps = np.asarray(group["n_steps"][is_success], dtype=float)
        mean_n_steps = float(np.mean(success_steps)) if n_success > 0 else np.nan
        std_n_steps = float(np.std(success_steps, ddof=1)) if n_success > 1 else np.nan
        cv_n_steps = std_n_steps / mean_n_steps if n_success > 1 and mean_n_steps > 0 else np.nan

        n_direction_changed = int(group["direction_changed"].sum())

        start_positions.append(group["start_position"][0])
        target_positions.append(group["target_position"][0])
        n_totals.append(n_total)
        n_successes.append(n_success)
        n_failures.append(n_total - n_success)
        success_rates.append(100.0 * n_success / n_total)
        mean_steps.append(mean_n_steps)
        std_steps.append(std_n_steps)
        cv_steps.append(cv_n_steps)
        n_dir_changed.append(n_direction_changed)
        dir_change_rates.append(100.0 * n_direction_changed / n_total)

    summary = Table({
        "start_position": start_positions,
        "target_position": target_positions,
        "n_total": n_totals,
        "n_success": n_successes,
        "n_failure": n_failures,
        "success_rate_pct": success_rates,
        "mean_n_steps_success": mean_steps,
        "std_n_steps_success": std_steps,
        "cv_n_steps_success": cv_steps,
        "n_direction_changed": n_dir_changed,
        "direction_change_rate_pct": dir_change_rates,
    })
    summary.sort(["start_position", "target_position"])
    return summary


def plot_steps_vs_success(table, output_path):
    """
    Plot success-vs-failure histograms of total_steps, zero_to_edge_steps, pos_to_zero_steps, ensure_zero_steps and their sum with one panel per start_position.

    Plot histograms of zero_to_edge_steps, pos_to_zero_steps, and
    ensure_zero_steps, with one panel per start_position. Color
    distinguishes the target direction (next vs previous position) and
    hatching distinguishes the outcome (success vs failure).

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).
    output_path : pathlib.Path
        Directory where the three figures ("zero_to_edge_steps.png",
        "pos_to_zero_steps.png", "ensure_zero_steps.png") are saved.
    """
    table["total_steps"] = table["zero_to_edge_steps"] + table["pos_to_zero_steps"] + table["ensure_zero_steps"]

    grouped = table[table["start_position"] != 0].group_by(["start_position"])

    
    movements = ['zero_to_edge_steps', 'pos_to_zero_steps', 'ensure_zero_steps', 'total_steps']   

    for movement in movements:
        fig, axes = plt.subplots(2, 4, figsize=(16,9))
        axes = axes.flatten()
    
        for i, group in enumerate(grouped.groups):
            ax = axes[i]

            start_position = group["start_position"][0]

            column = group[movement]
            valid = ~np.asarray(column.mask) if hasattr(column, "mask") else np.ones(len(column), dtype=bool)

            target_position = np.asarray(group["target_position"])[valid]
            steps = np.asarray(column[valid], dtype=float)

            is_success = np.asarray(group["outcome"] == "success")[valid]
            is_next = target_position == start_position + 1 
            is_previous = target_position == start_position - 1  


            bins = np.histogram_bin_edges(steps, bins="auto")
            ax.hist(steps[is_next & is_success], bins=bins, color="tab:blue", alpha=0.6,
                    edgecolor='black', rwidth=0.85, label="next position - success")
            ax.hist(steps[is_next & ~is_success], bins=bins, color="tab:blue", hatch="//", 
                    alpha=0.6, edgecolor='black', rwidth=0.85, label="next position - failure")
            ax.hist(steps[is_previous & is_success], bins=bins, color="tab:orange", alpha=0.6, 
                    edgecolor='black', rwidth=0.85, label="previous position - success")
            ax.hist(steps[is_previous & ~is_success], bins=bins, color="tab:orange", hatch="//", 
                    alpha=0.6, edgecolor='black', rwidth=0.85, label="previous position - failure")

            ax.set_title(f"Starting position {start_position}", fontsize=9)
            ax.set_xlabel("Steps", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

        axes[-1].axis("off")

        fig.tight_layout()
        fig.savefig(output_path / f'{movement}.png', dpi=300)
        plt.close(fig)


def compute_failure_conditions(table):
    """
    Compare failure rate and failure_reason breakdown across categorical
    conditions that may make a GOTO command more likely to fail, or more
    likely to fail for a specific reason: whether it was issued right
    after a server restart, whether the motor reversed direction during
    it, and the target position.

    For each condition, in addition to the overall failure_rate_pct,
    every failure_reason ("server_restart", "engage_failed",
    "second_attempt_failed") gets its own count (n_<reason>) and its
    share of that condition's failures (pct_<reason>, relative to
    n_failure, not n_total). n and pct are kept side by side because the
    per-reason counts can be small, especially for engage_failed and
    second_attempt_failed, and a percentage alone would be misleading.

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).

    Returns
    -------
    astropy.table.Table
        One row per condition, with n_total, n_failure, failure_rate_pct,
        and n_<reason>/pct_<reason> for each failure_reason. The
        "overall" row is the baseline for comparison.
    """
    recovered = np.asarray(table["recovered_after_restart"])
    reversed_direction = np.asarray(table["direction_changed"])

    conditions = [
        ("overall", np.ones(len(table), dtype=bool)),
        ("recovered_after_restart = True", recovered),
        ("recovered_after_restart = False", ~recovered),
        ("direction_changed = True", reversed_direction),
        ("direction_changed = False", ~reversed_direction),
        ("target position = 1", np.asarray(table["target_position"] == 1)),
        ("target position = 2", np.asarray(table["target_position"] == 2)),
        ("target position = 3", np.asarray(table["target_position"] == 3)),
        ("target position = 4", np.asarray(table["target_position"] == 4)),
        ("target position = 5", np.asarray(table["target_position"] == 5)),
        ("target position = 6", np.asarray(table["target_position"] == 6)),
        ("target position = 7", np.asarray(table["target_position"] == 7)),
    ]

    failure_reasons = ["server_restart", "engage_failed", "second_attempt_failed"]

    labels, n_totals, n_failures, failure_rates = [], [], [], []
    reason_n = {reason: [] for reason in failure_reasons}
    reason_pct = {reason: [] for reason in failure_reasons}

    for label, mask in conditions:
        subset = table[mask]
        n_total = len(subset)
        is_failure = np.asarray(subset["outcome"] == "failure")
        n_failure = int(is_failure.sum())

        labels.append(label)
        n_totals.append(n_total)
        n_failures.append(n_failure)
        failure_rates.append(100.0 * n_failure / n_total if n_total > 0 else np.nan)

        failure_subset = subset[is_failure]
        for reason in failure_reasons:
            n_reason = int((failure_subset["failure_reason"] == reason).sum())
            reason_n[reason].append(n_reason)
            reason_pct[reason].append(100.0 * n_reason / n_failure if n_failure > 0 else np.nan)

    columns = {
        "condition": labels,
        "n_total": n_totals,
        "n_failure": n_failures,
        "failure_rate_pct": failure_rates,
    }
    for reason in failure_reasons:
        columns[f"n_{reason}"] = reason_n[reason]
        columns[f"pct_{reason}"] = reason_pct[reason]

    return Table(columns)


def plot_step_histograms(table, output_path):
    """
    Plot a grid of n_steps histograms, one panel per (start_position,
    target_position) pair, using successful commands only.

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).
    output_path : pathlib.Path
        Path where the figure is saved.
    """
    success = table[table["outcome"] == "success"]
    grouped = success.group_by(["start_position", "target_position"])
    pairs = [(g["start_position"][0], g["target_position"][0], np.asarray(g["n_steps"])) for g in grouped.groups]
    pairs.sort(key=lambda item: (item[0], item[1]))

    n_pairs = len(pairs)
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)

    for panel_index, (start_position, target_position, steps) in enumerate(pairs):
        ax = axes[panel_index // n_cols, panel_index % n_cols]
        ax.hist(steps, bins="auto", color="tab:blue", edgecolor="black")
        ax.set_title(f"{start_position} \u2192 {target_position}  (n={len(steps)})", fontsize=9)
        ax.set_xlabel("n_steps", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused panels when the number of pairs does not fill the grid.
    for panel_index in range(n_pairs, n_rows * n_cols):
        axes[panel_index // n_cols, panel_index % n_cols].axis("off")

    fig.suptitle("Step count distribution per position pair (successful commands only)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_rotation_time_summary(table):
    """
    Summarize rotation time statistics separately for successful and failed commands.

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).

    Returns
    -------
    astropy.table.Table
        One row per outcome ("success", "failure"), with columns:
        outcome, n_total, mean_rotation_time, std_rotation_time.
    """
    complete = table[table["outcome"] != "incomplete"]

    outcomes, n_totals, means, medians, stds = [], [], [], [], []
    for outcome in ["success", "failure"]:
        values = np.asarray(complete["rotation_time"][complete["outcome"] == outcome], dtype=float)
        outcomes.append(outcome)
        n_totals.append(values.size)
        means.append(np.mean(values) if values.size > 0 else np.nan)
        medians.append(np.median(values) if values.size > 0 else np.nan)
        stds.append(np.std(values) if values.size > 1 else np.nan)

    return Table({
        "outcome": outcomes,
        "n_total": n_totals,
        "mean_rotation_time": means,
        "median_rotation_time": medians,
        "std_rotation_time": stds,
    })


def plot_rotation_time_histogram(table, output_path):
    """
    Plot a histogram of rotation_time, split by outcome

    Parameters
    ----------
    table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).
    output_path : pathlib.Path
        Path where the figure is saved.
    """
    
    complete = table[table["outcome"] != "incomplete"]
    is_success = np.asarray(complete["outcome"] == "success")
    values = np.asarray(complete["rotation_time"], dtype=float)

    success_values = values[is_success]
    failure_values = values[~is_success]

    # Shared bin edges so success and failure bars line up on the same axis.
    bins = np.histogram_bin_edges(values, bins="auto")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(success_values, bins=bins, color="tab:blue", alpha=0.6,
                edgecolor="black", label="success")
    ax.hist(failure_values, bins=bins, color="tab:red", alpha=0.6,
                edgecolor="black", label="failure")

    ax.set_xlabel("Rotation time [s]")
    ax.set_ylabel("Count")
    ax.set_title("HWP rotation time distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _wilson_interval(n_success, n_total, confidence=0.95):
    """
    Wilson score confidence interval for a binomial proportion.

    Preferred over the normal approximation because several cells here
    have low counts (fine granularity by position and by condition),
    where the normal approximation becomes unreliable or can even fall
    outside [0, 1].

    Parameters
    ----------
    n_success : int
        Number of successes for the proportion being estimated (e.g.
        number of failures, if estimating a failure rate).
    n_total : int
        Total number of trials.
    confidence : float, optional
        Confidence level, default 0.95.

    Returns
    -------
    tuple of float
        (center, lower, upper) Wilson-adjusted proportion and bounds.
        (nan, nan, nan) if n_total is zero.
    """
    if n_total == 0:
        return np.nan, np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return center, center - half_width, center + half_width


def _fisher_test_binary_split(is_group, is_success):
    """
    Compare failure rate between a binary grouping and its complement
    using Fisher's exact test on the resulting 2x2 table.

    Used both for one-vs-rest tests on multi-level categorical variables
    (e.g. a single position vs all others) and for natural two-level
    variables (booleans, or a "stage repeated" indicator).

    Parameters
    ----------
    is_group : numpy.ndarray of bool
        True for rows belonging to the group under test.
    is_success : numpy.ndarray of bool
        True for rows with outcome "success".

    Returns
    -------
    dict
        odds_ratio, p_value, n_success_in, n_failure_in, n_success_out,
        n_failure_out, failure_rate_in_pct, failure_rate_out_pct.
    """
    n_success_in = int(np.sum(is_group & is_success))
    n_failure_in = int(np.sum(is_group & ~is_success))
    n_success_out = int(np.sum(~is_group & is_success))
    n_failure_out = int(np.sum(~is_group & ~is_success))
    contingency_table = [[n_failure_in, n_success_in], [n_failure_out, n_success_out]]
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    n_in = n_success_in + n_failure_in
    n_out = n_success_out + n_failure_out
    return {
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "n_success_in": n_success_in,
        "n_failure_in": n_failure_in,
        "n_success_out": n_success_out,
        "n_failure_out": n_failure_out,
        "failure_rate_in_pct": 100.0 * n_failure_in / n_in if n_in > 0 else np.nan,
        "failure_rate_out_pct": 100.0 * n_failure_out / n_out if n_out > 0 else np.nan,
    }


def _mannwhitney_with_rank_biserial(success_values, failure_values):
    """
    Compare a continuous variable between success and failure groups
    with the Mann-Whitney U test, reporting rank-biserial correlation
    as effect size.

    Mann-Whitney is used instead of a t-test because step counts are
    expected to be right-skewed rather than normally distributed.

    Parameters
    ----------
    success_values : numpy.ndarray
        Values for outcome == "success" (masked/invalid entries already
        excluded by the caller).
    failure_values : numpy.ndarray
        Values for outcome == "failure" (masked/invalid entries already
        excluded by the caller).

    Returns
    -------
    dict
        statistic (U), p_value, rank_biserial, n_success, n_failure.
        Positive rank_biserial means the success group tends to have
        larger values. nan fields if either group is empty.
    """
    n_success = len(success_values)
    n_failure = len(failure_values)
    if n_success == 0 or n_failure == 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "rank_biserial": np.nan,
            "n_success": n_success,
            "n_failure": n_failure,
        }
    u_statistic, p_value = stats.mannwhitneyu(success_values, failure_values, alternative="two-sided")
    rank_biserial = 1 - (2 * u_statistic) / (n_success * n_failure)
    return {
        "statistic": u_statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "n_success": n_success,
        "n_failure": n_failure,
    }


def _benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction applied jointly to a set of
    p-values from parallel hypothesis tests.

    Parameters
    ----------
    p_values : array_like
        p-values to correct together. NaN entries are passed through
        unchanged and excluded from the correction.

    Returns
    -------
    numpy.ndarray
        FDR-corrected p-values, same order and length as the input.
    """
    p_values = np.asarray(p_values, dtype=float)
    corrected = np.full_like(p_values, np.nan)
    valid = ~np.isnan(p_values)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return corrected
    valid_idx = np.flatnonzero(valid)
    sorted_idx = valid_idx[np.argsort(p_values[valid_idx])]
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, n_valid + 1)
    adjusted = sorted_p * n_valid / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    corrected[sorted_idx] = np.clip(adjusted, 0, 1)
    return corrected


def compute_outcome_association_screening(commands_table):
    """
    Univariate screening of variables associated with command outcome.

    Tests, excluding "incomplete" commands: one-vs-rest Fisher exact
    tests for start_position and target_position; Fisher exact tests
    for the boolean variables direction_changed and
    recovered_after_restart; Mann-Whitney U tests (with rank-biserial
    effect size) for n_steps and the three masked step-count columns.
    For the masked columns, the "stage repeated" indicator (whether the
    value is masked) is also tested with Fisher exact, since the
    missingness itself flags a failed positioning attempt and is
    informative rather than incidental. All p-values are jointly
    corrected with Benjamini-Hochberg FDR.

    Parameters
    ----------
    commands_table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).

    Returns
    -------
    astropy.table.Table
        One row per test. Columns: variable, test, statistic, p_value,
        p_value_fdr, effect_size, n_success, n_failure. Sorted by
        p_value_fdr ascending.
    """
    complete = commands_table[commands_table["outcome"] != "incomplete"]
    is_success = np.asarray(complete["outcome"] == "success")

    variables, tests, statistics, p_values, effect_sizes = [], [], [], [], []
    n_successes, n_failures = [], []

    def _append_fisher_row(variable, result):
        variables.append(variable)
        tests.append("fisher_exact")
        statistics.append(result["odds_ratio"])
        p_values.append(result["p_value"])
        effect_sizes.append(result["failure_rate_in_pct"] - result["failure_rate_out_pct"])
        n_successes.append(result["n_success_in"] + result["n_success_out"])
        n_failures.append(result["n_failure_in"] + result["n_failure_out"])

    # Categorical variables: one-vs-rest Fisher exact test per position.
    for var_name in ["start_position", "target_position"]:
        values = np.asarray(complete[var_name])
        for position in np.unique(values):
            result = _fisher_test_binary_split(values == position, is_success)
            # only the "in-group" counts are meaningful for a single category
            variables.append(f"{var_name}=={position}")
            tests.append("fisher_exact_one_vs_rest")
            statistics.append(result["odds_ratio"])
            p_values.append(result["p_value"])
            effect_sizes.append(result["failure_rate_in_pct"] - result["failure_rate_out_pct"])
            n_successes.append(result["n_success_in"])
            n_failures.append(result["n_failure_in"])

    # Boolean variables: Fisher exact test on the natural True/False split.
    for var_name in ["direction_changed", "recovered_after_restart"]:
        values = np.asarray(complete[var_name], dtype=bool)
        _append_fisher_row(var_name, _fisher_test_binary_split(values, is_success))

    # Continuous variables: Mann-Whitney U on non-masked values.
    continuous_vars = ["n_steps", "pos_to_zero_steps", "ensure_zero_steps", "zero_to_edge_steps"]
    for var_name in continuous_vars:
        column = complete[var_name]
        valid = ~np.asarray(column.mask) if hasattr(column, "mask") else np.ones(len(column), dtype=bool)
        values = np.asarray(column, dtype=float)
        result = _mannwhitney_with_rank_biserial(values[valid & is_success], values[valid & ~is_success])
        variables.append(var_name)
        tests.append("mannwhitneyu")
        statistics.append(result["statistic"])
        p_values.append(result["p_value"])
        effect_sizes.append(result["rank_biserial"])
        n_successes.append(result["n_success"])
        n_failures.append(result["n_failure"])

        # Informative missingness: whether the stage was repeated after
        # a failed attempt to lock onto the target position.
        if var_name != "n_steps":
            _append_fisher_row(f"{var_name}_is_masked", _fisher_test_binary_split(~valid, is_success))

    p_values_fdr = _benjamini_hochberg(p_values)

    summary = Table({
        "variable": variables,
        "test": tests,
        "statistic": statistics,
        "p_value": p_values,
        "p_value_fdr": p_values_fdr,
        "effect_size": effect_sizes,
        "n_success": n_successes,
        "n_failure": n_failures,
    })
    summary.sort("p_value_fdr")
    return summary


def plot_outcome_association_screening(commands_table, save_path=None):
    """
    Plot outcome association for each screened variable: Wilson-CI
    failure-rate bars for categorical/boolean variables, box plots
    (success vs failure) for continuous variables.

    Parameters
    ----------
    commands_table : astropy.table.Table
        Combined per-command table (as returned by load_combined_csv).
    save_path : pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one subplot per variable.
    """
    complete = commands_table[commands_table["outcome"] != "incomplete"]
    is_success = np.asarray(complete["outcome"] == "success")

    categorical_specs = [
        ("start_position", np.asarray(complete["start_position"])),
        ("target_position", np.asarray(complete["target_position"])),
        ("direction_changed", np.asarray(complete["direction_changed"], dtype=bool)),
        ("recovered_after_restart", np.asarray(complete["recovered_after_restart"], dtype=bool)),
    ]
    continuous_specs = ["n_steps", "pos_to_zero_steps", "ensure_zero_steps", "zero_to_edge_steps"]

    n_panels = len(categorical_specs) + len(continuous_specs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    panel_index = 0

    for var_name, values in categorical_specs:
        ax = axes[panel_index]
        panel_index += 1
        categories = np.unique(values)
        centers, err_low, err_high = [], [], []
        for category in categories:
            mask = values == category
            n_total = int(np.sum(mask))
            n_failure = int(np.sum(mask & ~is_success))
            center, lower, upper = _wilson_interval(n_failure, n_total)
            centers.append(100 * center)
            err_low.append(100 * (center - lower))
            err_high.append(100 * (upper - center))
        x_pos = np.arange(len(categories))
        ax.bar(x_pos, centers, yerr=[err_low, err_high], capsize=4, color="tab:red", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(c) for c in categories])
        ax.set_ylabel("Failure rate [%]")
        ax.set_title(var_name)

    for var_name in continuous_specs:
        ax = axes[panel_index]
        panel_index += 1
        column = complete[var_name]
        valid = ~np.asarray(column.mask) if hasattr(column, "mask") else np.ones(len(column), dtype=bool)
        values = np.asarray(column, dtype=float)
        ax.boxplot([values[valid & is_success], values[valid & ~is_success]], labels=["success", "failure"])
        ax.set_ylabel(var_name)
        ax.set_title(var_name)

    for ax in axes[panel_index:]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the combined HWP GOTO command table: step repeatability, "
        "backslash, failure conditions, outcome screening, and restart diagnosis."
    )
    parser.add_argument("combined_csv", type=Path, help="Path to hwp_commands_combined.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary tables and figures are saved. Defaults to the input file's directory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.combined_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_combined_csv(args.combined_csv)

    position_pair_summary = compute_position_pair_summary(table)
    position_pair_summary.write(output_dir / "position_pair_summary.csv", format="csv", overwrite=True)

    failure_conditions = compute_failure_conditions(table)
    failure_conditions.write(output_dir / "failure_conditions.csv", format="csv", overwrite=True)

    plot_step_histograms(table, output_dir / "step_histograms_grid.png")
    plot_steps_vs_success(table, output_dir)

    screening = compute_outcome_association_screening(table)
    screening.write(output_dir / "outcome_screening.csv", format="csv", overwrite=True)

    restart_diagnosis = compute_restart_diagnosis(table)
    restart_diagnosis.write(output_dir / "restart_diagnosis.csv", format="csv", overwrite=True)
    restart_diagnosis_summary = summarize_restart_diagnosis(restart_diagnosis)
    restart_diagnosis_summary.write(output_dir / "restart_diagnosis_summary.csv", format="csv", overwrite=True)

    plot_outcome_association_screening(table, output_dir / "outcome_screening.png")

    rotation_time = compute_rotation_time_summary(table)
    rotation_time.write(output_dir / "rotation_time.csv", format="csv", overwrite=True)
    plot_rotation_time_histogram(table, output_dir / "rotation_time.png")

    print(f"Analyzed {len(table)} commands across {len(set(table['date']))} day(s).\n")

    #print("Position pair summary:")
    #position_pair_summary.pprint_all()

    print("\nFailure conditions:")
    failure_conditions.pprint_all()

    print("\nRestart diagnosis report:")
    restart_diagnosis_summary.pprint_all()

    print("\nRotation time report:")
    rotation_time.pprint_all()

    print(f"\nFigures and tables saved to {output_dir}")


if __name__ == "__main__":
    main()
