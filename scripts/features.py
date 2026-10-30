"""Load cached features and define four cumulative ablation configurations."""

from pathlib import Path

import numpy as np
import pandas as pd


LITE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = LITE_DIR / "features.parquet"

# Raw trades cover May 30 through October 25, inclusive, in UTC.
DATA_END = pd.Timestamp("2026-10-26 00:00:00", tz="UTC")

ACTIVITY = [
    "log_vol24", "log_vol6", "log_vol72", "fills24", "log_wallets24",
    "buy_share24", "avg_fill24", "peak_share24", "log_cum_vol",
]
ACCELERATION = ["share6", "share1", "growth6", "growth24"]
PRICE = ["pf", "d_pf_1", "d_pf_6", "d_pf_24", "range24", "dist_bound"]
CONTEXT = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "log_age",
    "log_h2end", "h2end_cap", "neg_risk", "sib_accel",
]
FEATURE_SETS = {
    "A: Activity": ACTIVITY,
    "AB: Activity + acceleration": ACTIVITY + ACCELERATION,
    "ABC: Activity + acceleration + price": ACTIVITY + ACCELERATION + PRICE,
    "ABCD: Activity + acceleration + price + context": ACTIVITY + ACCELERATION + PRICE + CONTEXT,
}


def engineer_features(raw):
    """Derive the 28 inputs from past aggregates; do not filter on future outcomes.

    Called by build_features.ipynb before its retrospective resolution exclusion.
    Price/outcome and amount conventions are inherited from the supplied hp table.
    """
    df = raw.drop(columns=["is_scheduled"], errors="ignore").copy()
    for column in ["h", "decision_time", "resolved_at"]:
        df[column] = pd.to_datetime(df[column], utc=True)

    # Keep offsets for growth ratios whose reference volume can be zero.
    df["share6"] = df.vol6 / (df.vol24 + 1)
    df["share1"] = df.vol / (df.vol24 + 1)
    # Helper for the existing baseline and peer feature; excluded from model inputs.
    df["accel6"] = (df.vol6 / 6) / (df.vol24 / 24 + 1)
    df["growth6"] = df.vol6 / (df.vol6_prev + 1)
    df["growth24"] = df.vol24 / (df.vol24_prev + 1)
    df["buy_share24"] = df.buy24 / (df.vol24 + 1)
    df["avg_fill24"] = df.vol24 / df.fills24.where(df.fills24 > 0)
    df["peak_share24"] = df.vmax24 / df.vol24.where(df.vol24 > 0)

    for hours in [1, 6, 24]:
        df[f"d_pf_{hours}"] = df.pf - df[f"pf_{hours}"]
    df["range24"] = (df.pmax24 - df.pmin24).fillna(0)
    df["dist_bound"] = np.minimum(df.pf, 1 - df.pf).fillna(0.5)

    for source in ["vol6", "vol24", "vol72", "wallets24", "cum_vol"]:
        df[f"log_{source}"] = np.log1p(df[source])
    df["log_age"] = np.log1p(df.age_h.clip(lower=0))
    df["log_h2end"] = np.log1p(df.hours_to_end.clip(0, 24 * 365))
    df["h2end_cap"] = df.hours_to_end.clip(-72, 24 * 90)

    hour = df.decision_time.dt.hour
    weekday = df.decision_time.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["neg_risk"] = df.neg_risk.fillna(False).astype(int)

    # Compute peer activity among still-open candidates before future-resolution filtering.
    peers = df.groupby(["event_slug", "h"])["accel6"]
    total = peers.transform("sum")
    count = peers.transform("count")
    df["sib_accel"] = ((total - df.accel6) / (count - 1).clip(lower=1)).where(count > 1)

    df["y"] = ((df.vol_fwd12 >= 10_000) & (df.vol_fwd12 > df.vol24)).astype(int)
    return df


def load_features(path=DATA_PATH):
    """Load the final local artifact created by build_features.ipynb."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run build_features.ipynb first."
        )
    # Ignore the retired title-pattern flag in older caches.
    df = pd.read_parquet(path).drop(columns=["is_scheduled"], errors="ignore")
    required = set(FEATURE_SETS[next(reversed(FEATURE_SETS))]) | {
        "market", "h", "decision_time", "y", "accel6", "resolved_at",
        "history_known_hours", "future_known_hours", "pipeline_version",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Not a rebuilt lite feature file; missing: {sorted(missing)}")
    for column in ["h", "decision_time", "resolved_at"]:
        df[column] = pd.to_datetime(df[column], utc=True)
    if not df.pipeline_version.eq("lite-v3").all():
        raise ValueError("Unexpected feature pipeline version; rebuild the local file.")
    if not (df.history_known_hours.eq(72) & df.future_known_hours.eq(12)).all():
        raise ValueError("Feature file contains incomplete observation windows.")
    if df.resolved_at.le(df.decision_time).any():
        raise ValueError("Feature file contains markets already closed at decision time.")
    if df.duplicated(["market", "h"]).any():
        raise ValueError("Duplicate market-hour rows in feature file.")
    if not df.decision_time.eq(df.h + pd.Timedelta(hours=1)).all():
        raise ValueError("Prediction time must be the end of the hourly bucket.")
    return df.sort_values(["decision_time", "market"]).reset_index(drop=True)
