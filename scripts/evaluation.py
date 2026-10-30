"""One chronological split and two evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


GAP = pd.Timedelta(hours=36)
VALID_START = pd.Timestamp("2026-09-01", tz="UTC")
TEST_START = pd.Timestamp("2026-10-01", tz="UTC")
TEST_END = pd.Timestamp("2026-10-26", tz="UTC")


def split_data(df):
    """Leave a 36-hour gap before validation and before testing."""
    t = df.decision_time
    train = df.loc[t < VALID_START - GAP].copy()
    valid = df.loc[(t >= VALID_START) & (t < TEST_START - GAP)].copy()
    test = df.loc[(t >= TEST_START) & (t < TEST_END)].copy()
    return train, valid, test


def split_summary(train, valid, test):
    """Show the actual observed dates rather than approximate month labels."""
    rows = []
    for name, part in [("Train", train), ("Validation", valid), ("Test", test)]:
        rows.append({
            "split": name,
            "first_decision_time_utc": part.decision_time.min(),
            "last_decision_time_utc": part.decision_time.max(),
            "rows": len(part),
            "markets": part.market.nunique(),
            "positive_rate": part.y.mean(),
        })
    return pd.DataFrame(rows)


def evaluate(y, scores):
    """AP measures ranking quality; precision@5% measures a fixed review budget."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(0.05 * len(y)))
    # Stable sorting makes ties reproducible in the fixed input row order.
    top = np.argsort(-scores, kind="stable")[:k]
    return {
        "AP": average_precision_score(y, scores),
        "precision_at_5pct": y[top].mean(),
    }
