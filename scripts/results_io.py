"""CSV persistence; no pricing or statistical calculations."""

import csv
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write dictionaries to CSV with a stable union of fields."""
    if not rows:
        raise ValueError("rows cannot be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_numeric_csv(path):
    """Read experiment records using explicit numeric columns, without guessing."""
    integer_fields = {
        "n_paths", "repetition", "seed", "ci_covers_exact", "n_strata",
        "payoff_evaluations", "pilot_evaluations", "qmc_replications",
    }
    float_fields = {
        "S0", "K", "r", "sigma", "T", "analytical_price", "mc_price",
        "signed_error", "absolute_error", "relative_error", "squared_error",
        "reported_std_error", "ci_lower", "ci_upper", "estimator_variance",
        "runtime_seconds", "beta", "parameter_value", "moneyness",
    }
    rows = read_csv(path)
    for row in rows:
        for key, value in row.items():
            if value != "" and key in integer_fields:
                row[key] = int(value)
            elif value != "" and key in float_fields:
                row[key] = float(value)
    return rows
