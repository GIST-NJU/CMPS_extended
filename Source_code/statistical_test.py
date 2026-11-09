"""Run RQ1 Wilcoxon tests from the extracted raw-metric CSV files.

Input files (all in Experiment_results/RQ1/statistical_analysis):
  - rs_nsga_raw_metrics.csv
  - mpss_raw_metrics.csv
  - cmps_cmpspp_raw_metrics.csv

CMPS++ and CMPS are deterministic under fixed experimental inputs.  Thus,
CMPS++ is treated as a fixed reference against the ten stochastic repetitions
of NSGA-II, RS, and MPSS.  The resulting one-sample signed-rank test on the
ten differences is numerically equivalent to a paired signed-rank test with a
fixed CMPS++ reference.  CMPS++ versus CMPS is reported descriptively only.

Run from the project root:
  python Source_code/rq1_statistical_test.py
  python Source_code/rq1_statistical_test.py --holm-scope global
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "Experiment_results" / "RQ1" / "statistical_analysis"
INPUTS = {
    "rs_nsga_raw_metrics.csv": {"RS", "NSGA-II"},
    "mpss_raw_metrics.csv": {"MPSS"},
    "cmps_cmpspp_raw_metrics.csv": {"CMPS", "CMPS++"},
}
BASELINES = ("CMPS", "NSGA-II", "MPSS", "RS")


def read_metrics() -> list[dict[str, Any]]:
    rows = []
    for filename, methods in INPUTS.items():
        with (RESULT_ROOT / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["Method"] not in methods:
                    continue
                rows.append({
                    "Method": row["Method"],
                    "Subject": row["Subject"],
                    "Budget": int(row["Budget"]),
                    "Run": int(row["Run"]),
                    "TRC": float(row["TRC"]),
                    "FDR": float(row["FDR"]),
                })
    return rows


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(values[order[start]], values[order[end]], abs_tol=1e-14):
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def signed_rank(differences: list[float]) -> tuple[float, float, int]:
    """Return exact two-sided p, rank-biserial r, and nonzero count."""
    nonzero = [value for value in differences if not math.isclose(value, 0.0, abs_tol=1e-14)]
    if not nonzero:
        return 1.0, 0.0, 0
    ranks = average_ranks([abs(value) for value in nonzero])
    w_plus = sum(rank for rank, difference in zip(ranks, nonzero) if difference > 0)
    total = sum(ranks)
    effect = (2.0 * w_plus - total) / total
    possible_w_plus = [
        sum(rank for rank, sign in zip(ranks, signs) if sign)
        for signs in itertools.product((0, 1), repeat=len(ranks))
    ]
    lower = sum(value <= w_plus + 1e-12 for value in possible_w_plus) / len(possible_w_plus)
    upper = sum(value >= w_plus - 1e-12 for value in possible_w_plus) / len(possible_w_plus)
    return min(1.0, 2.0 * min(lower, upper)), effect, len(nonzero)


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    tested = [row for row in rows if row["p_value"] is not None]
    for row in rows:
        row["p_value_holm"] = None
        row["significant_holm_0.05"] = "not_applicable"
    ordered = sorted(tested, key=lambda row: row["p_value"])
    previous = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * row["p_value"])
        previous = max(previous, adjusted)
        row["p_value_holm"] = previous
        row["significant_holm_0.05"] = previous < 0.05


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_tables(rows: list[dict[str, Any]]) -> None:
    """Create two complete LaTeX tables with adjusted p-values and effect sizes."""
    output = RESULT_ROOT / "rq1_statistical_tables.tex"
    configurations = sorted({(row["Subject"], int(row["Budget"])) for row in rows})
    blocks = []
    for metric in ("TRC", "FDR"):
        lines = [
            "\\begin{table*}[t]",
            f"\\caption{{Statistical results for {metric}. Each cell reports the Holm--Bonferroni-adjusted $p$-value / rank-biserial correlation $r$. Positive $r$ favors CMPS\\texttt{{++}}.}}",
            "\\centering",
            "\\scriptsize",
            "\\begin{tabular}{l c c c c}",
            "\\toprule",
            "\\textbf{Subject} & $k$ & \\textbf{NSGA-II} & \\textbf{MPSS} & \\textbf{RS} \\\\",
            "\\midrule",
        ]
        for subject, budget in configurations:
            values = []
            for baseline in ("NSGA-II", "MPSS", "RS"):
                row = next(item for item in rows if item["Subject"] == subject and
                           int(item["Budget"]) == budget and item["Metric"] == metric and
                           item["Baseline"] == baseline)
                values.append(f"${float(row['P_Value_Holm']):.4f} / {float(row['Rank_Biserial_r']):.2f}$")
            lines.append(f"{subject.replace('_', '\\_')} & {budget} & " + " & ".join(values) + " \\\\")
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\label{{tab:stat_{metric.lower()}}}",
            "\\end{table*}",
        ])
        blocks.append("\n".join(lines))
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Wilcoxon signed-rank analysis.")
    parser.add_argument("--holm-scope", choices=("per-configuration", "global"), default="per-configuration")
    args = parser.parse_args()

    raw_rows = read_metrics()
    write_csv(RESULT_ROOT / "rq1_all_raw_metrics.csv", raw_rows,
              ["Method", "Subject", "Budget", "Run", "TRC", "FDR"])

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["Method"], row["Subject"], row["Budget"])].append(row)

    statistics = []
    configurations = sorted({(row["Subject"], row["Budget"]) for row in raw_rows})
    for subject, budget in configurations:
        cmpspp = grouped[("CMPS++", subject, budget)]
        if len(cmpspp) != 1:
            raise ValueError(f"CMPS++ must have one deterministic value for {subject}, k={budget}.")
        for baseline in BASELINES:
            baseline_rows = grouped[(baseline, subject, budget)]
            for metric in ("TRC", "FDR"):
                reference = cmpspp[0][metric]
                values = [row[metric] for row in baseline_rows]
                result: dict[str, Any] = {
                    "Subject": subject,
                    "Budget": budget,
                    "Metric": metric,
                    "Baseline": baseline,
                    "CMPSpp_Value": reference,
                    "Baseline_Mean": sum(values) / len(values),
                    "Difference_PP": (reference - sum(values) / len(values)) * 100.0,
                    "Baseline_Runs": len(values),
                    "Test": "not_applicable",
                    "P_Value": None,
                    "P_Value_Holm": None,
                    "Rank_Biserial_r": None,
                    "Nonzero_Differences": None,
                    "Significant_Holm_0.05": "not_applicable",
                    "Note": "",
                }
                if baseline == "CMPS":
                    result["Note"] = "Both CMPS++ and CMPS are deterministic; reported descriptively."
                else:
                    differences = [reference - value for value in values]
                    p_value, effect, nonzero = signed_rank(differences)
                    result.update({
                        "Test": "one_sample_wilcoxon_signed_rank",
                        "P_Value": p_value,
                        "Rank_Biserial_r": effect,
                        "Nonzero_Differences": nonzero,
                        "Note": "CMPS++ is the fixed deterministic reference; positive r favors CMPS++.",
                    })
                statistics.append(result)

    # Internal keys make Holm adjustment easier; they are removed before export.
    for row in statistics:
        row["p_value"] = row["P_Value"]
    if args.holm_scope == "global":
        for metric in ("TRC", "FDR"):
            holm_adjust([row for row in statistics if row["Metric"] == metric])
    else:
        families: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in statistics:
            families[(row["Subject"], row["Budget"])].append(row)
        for family in families.values():
            holm_adjust(family)
    for row in statistics:
        row["P_Value_Holm"] = row.pop("p_value_holm")
        row["Significant_Holm_0.05"] = row.pop("significant_holm_0.05")
        row.pop("p_value")
        for field in ("CMPSpp_Value", "Baseline_Mean", "Difference_PP", "P_Value", "P_Value_Holm", "Rank_Biserial_r"):
            if isinstance(row[field], float):
                row[field] = f"{row[field]:.10f}"
    fields = ["Subject", "Budget", "Metric", "Baseline", "CMPSpp_Value", "Baseline_Mean", "Difference_PP",
              "Baseline_Runs", "Test", "P_Value", "P_Value_Holm", "Rank_Biserial_r", "Nonzero_Differences",
              "Significant_Holm_0.05", "Note"]
    write_csv(RESULT_ROOT / "rq1_wilcoxon_statistics.csv", statistics, fields)
    write_latex_tables(statistics)
    print(f"Saved raw metrics to {RESULT_ROOT / 'rq1_all_raw_metrics.csv'}")
    print(f"Saved statistics to {RESULT_ROOT / 'rq1_wilcoxon_statistics.csv'}")
    print(f"Saved LaTeX tables to {RESULT_ROOT / 'rq1_statistical_tables.tex'}")
    print(f"Holm correction scope: {args.holm_scope}")


if __name__ == "__main__":
    main()
