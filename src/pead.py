"""Price-conditioned post-earnings announcement drift.

The classical PEAD result conditions on an earnings surprise: the gap between
reported earnings and what analysts expected. No expectation data was supplied
and none is needed to run the test the brief asks for, but the distinction
matters for what may be claimed. Conditioning on the day-zero price move asks a
different and weaker question -- does the market keep moving the way it first
moved -- and a positive result here would not establish under-reaction to a
surprise. The naming throughout says price-conditioned for that reason.

Construction:

  D0        the first session in which the earnings could be acted upon.
  R0        P_pre through the D0 close. For a release that first becomes
            tradable at the open this runs from the previous session's close,
            so the overnight gap is part of the initial reaction.
  direction sign(R0), the market's own reading of the news.
  drift_k   D0 close through the close of session D0+k, signed by direction.
            Positive means the move continued, negative means it reversed.

Under-reaction is the hypothesis under test, not the expected answer. With 57
events the honest output may well be that nothing is distinguishable from
normal variation, so every horizon is reported with the smallest effect this
sample could have detected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats_tools as st

PEAD_HORIZONS = [1, 3, 5, 10, 20]


def build_pead_events(events: pd.DataFrame) -> pd.DataFrame:
    """Select earnings events and attach signed drift at every horizon."""
    df = events[
        (events["subject"] == "Financial Results")
        & (events["exclusion_reason"] == "")
        & events["r_to_close"].notna()
    ].copy()

    df["R0"] = df["r_to_close"]
    df["R0_adj"] = df["adj_r_to_close"]
    df["direction"] = np.sign(df["R0"]).replace(0, np.nan)

    for h in PEAD_HORIZONS:
        df[f"signed_drift_{h}d"] = df["direction"] * df[f"drift_{h}d"]
        df[f"signed_adj_drift_{h}d"] = df["direction"] * df[f"adj_drift_{h}d"]

    df["block_m"] = st.block_label(df, 20)
    return df


def _horizon_row(df: pd.DataFrame, col: str, horizon: int, scope: str, measure: str) -> dict:
    s = df[col].dropna()
    summ = st.one_sample_summary(s, block_df=df.dropna(subset=[col]), value_col=col, block_col="block_m")
    return {
        "scope": scope,
        "measure": measure,
        "horizon_sessions": horizon,
        "n_events": summ["n"],
        "mean_drift": summ["mean"],
        "median_drift": summ["median"],
        "sd": summ["sd"],
        "continuation_rate": summ.get("share_positive"),
        "n_continued": summ.get("n_positive"),
        "ci_low_blocked": summ["ci_low"],
        "ci_high_blocked": summ["ci_high"],
        "ci_low_naive": summ["ci_low_naive"],
        "ci_high_naive": summ["ci_high_naive"],
        "p_ttest": summ["p_ttest"],
        "p_wilcoxon": summ["p_wilcoxon"],
        "p_sign_vs_50pct": summ["p_sign"],
        "min_detectable_effect": st.minimum_detectable_effect(summ["n"], summ["sd"]),
        "significant_blocked": bool(
            np.isfinite(summ["ci_low"]) and np.isfinite(summ["ci_high"])
            and (summ["ci_low"] > 0 or summ["ci_high"] < 0)
        ),
    }


def summarise(pead: pd.DataFrame) -> pd.DataFrame:
    """Drift by horizon, pooled and by stock, raw and peer-adjusted."""
    rows: list[dict] = []

    for measure, prefix in [("raw", "signed_drift"), ("peer_adjusted", "signed_adj_drift")]:
        for h in PEAD_HORIZONS:
            rows.append(_horizon_row(pead, f"{prefix}_{h}d", h, "POOLED", measure))
        for sym, g in pead.groupby("symbol"):
            for h in PEAD_HORIZONS:
                rows.append(_horizon_row(g, f"{prefix}_{h}d", h, sym, measure))

    # The initial reaction itself, for scale comparison against the drift.
    for measure, col in [("raw", "R0"), ("peer_adjusted", "R0_adj")]:
        s = pead[col].dropna()
        rows.append({
            "scope": "POOLED", "measure": measure, "horizon_sessions": 0,
            "n_events": len(s), "mean_drift": s.mean(), "median_drift": s.median(),
            "sd": s.std(ddof=1),
            "continuation_rate": np.nan, "n_continued": np.nan,
            "ci_low_blocked": np.nan, "ci_high_blocked": np.nan,
            "ci_low_naive": np.nan, "ci_high_naive": np.nan,
            "p_ttest": np.nan, "p_wilcoxon": np.nan, "p_sign_vs_50pct": np.nan,
            "min_detectable_effect": np.nan, "significant_blocked": False,
        })
        rows.append({
            "scope": "POOLED", "measure": measure, "horizon_sessions": -1,
            "n_events": len(s), "mean_drift": s.abs().mean(), "median_drift": s.abs().median(),
            "sd": s.abs().std(ddof=1),
            "continuation_rate": np.nan, "n_continued": np.nan,
            "ci_low_blocked": np.nan, "ci_high_blocked": np.nan,
            "ci_low_naive": np.nan, "ci_high_naive": np.nan,
            "p_ttest": np.nan, "p_wilcoxon": np.nan, "p_sign_vs_50pct": np.nan,
            "min_detectable_effect": np.nan, "significant_blocked": False,
        })

    out = pd.DataFrame(rows)
    out.loc[out["horizon_sessions"] == 0, "measure"] += " | initial reaction R0 (signed)"
    out.loc[out["horizon_sessions"] == -1, "measure"] += " | initial reaction |R0|"
    return out.sort_values(["measure", "scope", "horizon_sessions"]).reset_index(drop=True)


def drift_persistence(summary: pd.DataFrame, measure: str = "raw") -> dict:
    """How long, if at all, the drift stays distinguishable from zero."""
    pooled = summary[
        (summary["scope"] == "POOLED")
        & (summary["measure"] == measure)
        & (summary["horizon_sessions"] > 0)
    ].sort_values("horizon_sessions")

    visible = pooled[pooled["significant_blocked"]]
    return {
        "last_significant_horizon": int(visible["horizon_sessions"].max()) if len(visible) else 0,
        "horizons_tested": pooled["horizon_sessions"].tolist(),
        "n_significant": int(len(visible)),
        "smallest_detectable_at_20d": float(
            pooled.loc[pooled["horizon_sessions"] == 20, "min_detectable_effect"].iloc[0]
        ) if (pooled["horizon_sessions"] == 20).any() else np.nan,
    }
