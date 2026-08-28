"""Sensitivity of the conclusions to the two arbitrary numbers in the study.

Every quantity here is either measured or derived, apart from two judgment
calls that were set by reasoning rather than by the data:

  the 36-hour clustering window, which decides how many independent events
  exist at all, and

  the 2.0x thresholds in the strong-impact rule, which decide which subjects
  qualify.

Both are defensible. Neither is forced. A conclusion that only holds at the
chosen value is a conclusion about the choice, so both are swept and the
results reported next to the headline ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cluster, config, impact, stats_tools as st

CLUSTER_WINDOWS = [12, 24, 36, 48, 72]
RETURN_MULTIPLES = [1.5, 2.0, 2.5, 3.0]
VOLUME_RATIOS = [1.5, 2.0, 2.5]


def clustering_window_sweep(merged: pd.DataFrame, bars: dict[str, pd.DataFrame],
                            windows=None) -> pd.DataFrame:
    """Rebuild events at several window lengths and re-test the main claims.

    The pipeline is re-run from clustering onwards for each window, so the
    returns and volume ratios are recomputed against the new event times rather
    than reused.
    """
    windows = windows or CLUSTER_WINDOWS
    original = config.CLUSTER_WINDOW_HOURS
    rows = []

    try:
        for hours in windows:
            config.CLUSTER_WINDOW_HOURS = hours
            events = cluster.collapse_to_events(cluster.build_clusters(merged))
            events = impact.measure(events, bars)
            ok = events[events["exclusion_reason"] == ""]

            kw = st.kruskal_by_group(ok, "subject", "abs_adj_r_60m")
            fr = ok[ok["subject"] == "Financial Results"]
            ow = ok[ok["subject"] == "Order Win"]
            rt = ok[ok["subject"] == "Routine Filing"]

            rows.append({
                "cluster_window_hours": hours,
                "n_events": len(ok),
                "filings_per_event": float(ok["n_filings"].sum() / len(ok)),
                "n_earnings_events": len(fr),
                "kw_p": kw["p"],
                "kw_epsilon_sq": kw["epsilon_sq"],
                "median_abs_r60_results": fr["abs_adj_r_60m"].median(),
                "median_vol_ratio_results": fr["vol_ratio_60m"].median(),
                "median_abs_r60_orderwin": ow["abs_adj_r_60m"].median(),
                "median_abs_r60_routine": rt["abs_adj_r_60m"].median(),
                "orderwin_share_positive": float((ow["adj_r_60m"] > 0).mean()),
                "is_headline_setting": hours == original,
            })
    finally:
        config.CLUSTER_WINDOW_HOURS = original

    return pd.DataFrame(rows)


def strong_impact_sweep(subject_summary: pd.DataFrame,
                        return_multiples=None, volume_ratios=None) -> pd.DataFrame:
    """Which subjects qualify as strong, across the threshold grid.

    Recomputed from the summary rather than from the events, because the two
    thresholds only enter at the verdict step.
    """
    return_multiples = return_multiples or RETURN_MULTIPLES
    volume_ratios = volume_ratios or VOLUME_RATIOS

    rows = []
    for rm in return_multiples:
        for vr in volume_ratios:
            passing = []
            for r in subject_summary.itertuples(index=False):
                base = r.baseline_abs_r_60m
                size = np.isfinite(base) and base > 0 and r.median_abs_adj_r_60m >= rm * base
                vol = r.median_vol_ratio_60m >= vr
                evidence = np.isfinite(r.abs_r_60m_ci_low) and np.isfinite(base) and r.abs_r_60m_ci_low > base
                if size and vol and evidence:
                    passing.append(r.subject)
            rows.append({
                "return_multiple": rm,
                "volume_ratio": vr,
                "n_subjects_strong": len(passing),
                "subjects_strong": ", ".join(sorted(passing)) or "(none)",
                "is_headline_setting": rm == config.STRONG_RETURN_MULTIPLE and vr == config.STRONG_VOLUME_RATIO,
            })
    return pd.DataFrame(rows)
