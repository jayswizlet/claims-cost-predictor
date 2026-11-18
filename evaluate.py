"""
Evaluate the trained model on the held-out set and choose an operating point.

Rebuilds the test split train.py held out, prints a threshold sweep with the
net dollar value of each threshold, and writes ROC, precision-recall and
calibration curves to reports/evaluation_curves.png.

    python evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_DIR = Path("reports")

# Must match train.py so the model is scored on rows it never saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

VALID_USE_CASES = ("care_management", "stop_loss", "pricing")

# TODO: set the use case, threshold and justification.
CHOSEN_USE_CASE = "TODO"
CHOSEN_THRESHOLD = None
THRESHOLD_JUSTIFICATION = ""

# Rough care-management economics, in dollars.
OUTREACH_COST_PER_FLAGGED = 250
SAVINGS_PER_TRUE_POSITIVE = 4_000


def load_holdout() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Rebuild train.py's held-out split and return (X_test, y_test, y_proba)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found — run `python train.py` first")
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    X = df[features]
    y = df["is_high_cost"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_proba


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """
    Confusion counts, precision and recall at each threshold (default 0.05 to
    0.95). A member is flagged when y_proba >= threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    rows = []
    for t in thresholds:
        flagged = y_proba >= t
        tp = int(np.sum(flagged & (y_true == 1)))
        fp = int(np.sum(flagged & (y_true == 0)))
        fn = int(np.sum(~flagged & (y_true == 1)))
        n_flagged = tp + fp
        # Nobody flagged means precision is 0/0; report 0.0 rather than nan.
        precision = tp / n_flagged if n_flagged else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "threshold": round(float(t), 2),
            "n_flagged": n_flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    return pd.DataFrame(rows)


def expected_value(tp: int, fp: int, outreach_cost: float, savings_per_tp: float) -> float:
    """Net dollars: savings from true positives minus outreach to everyone flagged."""
    return tp * savings_per_tp - (tp + fp) * outreach_cost


def plot_curves(y_true, y_proba, out_path: Path = REPORT_DIR / "evaluation_curves.png") -> Path:
    """
    Save ROC, precision-recall and calibration curves side by side as one PNG,
    marking CHOSEN_THRESHOLD on the first two. Returns the path written.
    """
    import matplotlib

    matplotlib.use("Agg")  # write files without needing a display
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    fig, (ax_roc, ax_pr, ax_cal) = plt.subplots(1, 3, figsize=(16, 5))

    ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    ax_roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")

    ax_pr.plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_proba):.3f}")
    ax_pr.axhline(y_true.mean(), linestyle="--", color="grey", label="base rate")
    ax_pr.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")

    # Mark the proposed operating point once one has been chosen.
    if CHOSEN_THRESHOLD is not None:
        flagged = y_proba >= CHOSEN_THRESHOLD
        tp = np.sum(flagged & (y_true == 1))
        fp = np.sum(flagged & (y_true == 0))
        op_tpr = tp / max(y_true.sum(), 1)
        op_fpr = fp / max((y_true == 0).sum(), 1)
        op_precision = tp / flagged.sum() if flagged.sum() else 0.0
        label = f"threshold = {CHOSEN_THRESHOLD}"
        ax_roc.scatter(op_fpr, op_tpr, color="red", zorder=5, label=label)
        ax_pr.scatter(op_tpr, op_precision, color="red", zorder=5, label=label)

    ax_cal.plot(prob_pred, prob_true, marker="o", label="model")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax_cal.set(title="Calibration (quantile bins)", xlabel="Mean predicted probability",
               ylabel="Observed high-cost rate")

    for ax in (ax_roc, ax_pr, ax_cal):
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    X_test, y_test, y_proba = load_holdout()
    print(f"Scored {len(y_test)} held-out members "
          f"({int(y_test.sum())} of them high-cost at ${HIGH_COST_THRESHOLD_USD:,})")

    sweep = threshold_sweep(y_test, y_proba)
    sweep = sweep.assign(
        net_dollars=[
            expected_value(row.tp, row.fp, OUTREACH_COST_PER_FLAGGED, SAVINGS_PER_TRUE_POSITIVE)
            for row in sweep.itertuples()
        ]
    )
    print("\n=== Threshold sweep ===")
    print(sweep.to_string(index=False))

    path = plot_curves(y_test, y_proba)
    print(f"\nCurves written to {path}")

    print(f"\n=== My operating point ===")
    print(f"Use case:      {CHOSEN_USE_CASE}")
    print(f"Threshold:     {CHOSEN_THRESHOLD}")
    print(f"Justification: {THRESHOLD_JUSTIFICATION}")


if __name__ == "__main__":
    main()
