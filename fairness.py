"""
Audit the model's decisions across demographic groups.

Reports per-group rates for sex and age_bucket, applies the four-fifths rule,
compares against a model retrained without the demographic columns, and writes
reports/fairness_report.md.

    python fairness.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_PATH = Path("reports/fairness_report.md")

# Must match train.py so the blind model is fitted on the same training rows.
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROTECTED_COLUMNS = ["sex", "age_bucket"]
DECISION_THRESHOLD = 0.2  # keep in sync with evaluate.CHOSEN_THRESHOLD

# TODO: write the assessment from this cohort's numbers.
ASSESSMENT = ""


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return {"tp", "fp", "fn", "tn"} counts for 0/1 labels and predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def group_metrics(df: pd.DataFrame, group_col: str,
                  y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """
    One row per group, sorted by group:

        n              members in the group
        base_rate      share of the group that is actually high-cost
        positive_rate  share of the group the model flags   (demographic parity)
        tpr            of the actually-high-cost, share flagged
        fpr            of the actually-normal, share flagged (equalized odds)
        fnr            1 - tpr; the members the model misses
        precision      of the flagged, share actually high-cost

    A rate with an empty denominator is nan, not 0.
    """
    def rate(num: int, den: int) -> float:
        # An empty denominator means the rate is undefined, not zero.
        return num / den if den else np.nan

    rows = []
    for group, part in df.groupby(group_col, sort=True):
        c = confusion_counts(part[y_true_col], part[y_pred_col])
        n = len(part)
        tpr = rate(c["tp"], c["tp"] + c["fn"])
        rows.append({
            "group": group,
            "n": n,
            "base_rate": rate(c["tp"] + c["fn"], n),
            "positive_rate": rate(c["tp"] + c["fp"], n),
            "tpr": tpr,
            "fpr": rate(c["fp"], c["fp"] + c["tn"]),
            "fnr": 1 - tpr,
            "precision": rate(c["tp"], c["tp"] + c["fp"]),
        })
    return pd.DataFrame(rows)


def disparate_impact_ratio(rates: pd.Series) -> float:
    """Lowest rate divided by highest, ignoring nan; 1.0 is exact parity."""
    known = pd.Series(rates).dropna()
    if known.empty or known.max() == 0:
        return np.nan
    return float(known.min() / known.max())


def passes_80_percent_rule(ratio: float) -> bool:
    """True when the ratio clears the EEOC four-fifths guideline (0.80 passes)."""
    return bool(ratio >= 0.80)


def blind_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrain the same pipeline without PROTECTED_COLUMNS on the same training
    split, then compare disparate impact ratios with and without them.

    `df` must already carry y_true and y_pred from the full model.
    """
    full = joblib.load(MODEL_PATH)
    features = list(full["preprocessor"].feature_names_in_)
    blind_features = [f for f in features if f not in PROTECTED_COLUMNS]

    # Same pipeline shape, minus the protected columns in each transformer.
    blind = clone(full)
    blind.set_params(preprocessor__transformers=[
        (name, transformer, [c for c in cols if c not in PROTECTED_COLUMNS])
        for name, transformer, cols in full["preprocessor"].transformers
        if [c for c in cols if c not in PROTECTED_COLUMNS]
    ])

    y = df["y_true"]
    X_train, _, y_train, _ = train_test_split(
        df[blind_features], y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    blind.fit(X_train, y_train)

    blind_scored = df.assign(
        y_pred=(blind.predict_proba(df[blind_features])[:, 1] >= DECISION_THRESHOLD).astype(int)
    )

    rows = []
    for col in PROTECTED_COLUMNS:
        with_demo = disparate_impact_ratio(group_metrics(df, col).set_index("group")["positive_rate"])
        without = disparate_impact_ratio(
            group_metrics(blind_scored, col).set_index("group")["positive_rate"]
        )
        rows.append({
            "protected_column": col,
            "di_ratio_with_demographics": round(with_demo, 3),
            "di_ratio_blind": round(without, 3),
            "passes_80_with": passes_80_percent_rule(with_demo),
            "passes_80_blind": passes_80_percent_rule(without),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    scored = df.assign(
        y_true=df["is_high_cost"].astype(int),
        y_proba=pipeline.predict_proba(df[features])[:, 1],
    )
    scored["y_pred"] = (scored["y_proba"] >= DECISION_THRESHOLD).astype(int)

    print(f"Auditing at threshold {DECISION_THRESHOLD} "
          f"(label: >${HIGH_COST_THRESHOLD_USD:,} plan-paid)")

    lines = [f"# Fairness audit\n", f"Decision threshold: {DECISION_THRESHOLD}\n"]
    for col in PROTECTED_COLUMNS:
        table = group_metrics(scored, col)
        ratio = disparate_impact_ratio(table.set_index("group")["positive_rate"])
        verdict = "PASS" if passes_80_percent_rule(ratio) else "FAIL"

        print(f"\n=== {col} ===")
        print(table.to_string(index=False))
        print(f"disparate impact ratio (positive rate): {ratio:.3f} → {verdict}")

        lines += [f"\n## {col}\n", table.to_markdown(index=False),
                  f"\n\nDisparate impact ratio: {ratio:.3f} ({verdict})\n"]

    print("\n=== Blind model comparison ===")
    comparison = blind_model_comparison(scored)
    print(comparison.to_string(index=False))
    lines += ["\n## Blind model comparison\n", comparison.to_markdown(index=False),
              f"\n\n## Assessment\n\n{ASSESSMENT}\n"]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
