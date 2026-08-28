"""Subject-level aggregation and the definition of a strong impact.

"Strong" needs a definition that can be checked rather than asserted, so three
conditions must all hold, and each one rules out a different way of being
fooled:

  size          the median absolute move must be at least STRONG_RETURN_MULTIPLE
                times what the same stock normally does over the same minutes of
                the day. This is relative to the stock's own noise, so a
                volatile small cap does not qualify simply for being volatile.

  confirmation  median volume must be at least STRONG_VOLUME_RATIO times its
                time-of-day baseline. A price move on no volume is usually a
                wide spread, not a repricing.

  evidence      the 95% block-bootstrap interval for the median absolute move
                must sit clear of the same group's own no-news baseline. This
                is what stops a large point estimate on twelve events from
                being read as a finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats_tools as st

RETURN_WINDOWS = ["r_5m", "r_30m", "r_60m", "r_to_close"] + [f"r_{h}d" for h in config.DAILY_HORIZONS]


def subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    """One row per subject with impact statistics and the strong-impact verdict."""
    df = events[events["exclusion_reason"] == ""].copy()
    df["block_d"] = st.block_label(df, 1)
    df["log_vol_ratio_60m"] = np.log(df["vol_ratio_60m"].replace(0, np.nan))

    rows = []
    for subj, g in df.groupby("subject"):
        rec: dict = {
            "subject": subj,
            "n_events": len(g),
            "n_filings": int(g["n_filings"].sum()),
            "n_stocks": g["symbol"].nunique(),
            "share_tradable_at_open": float((g["tradable_regime"] == "at_open").mean()),
            "share_overlapping_60m": float(g["overlap_60m"].mean()),
        }

        for col in RETURN_WINDOWS:
            adj = f"adj_{col}"
            rec[f"mean_{col}"] = g[col].mean()
            rec[f"median_{col}"] = g[col].median()
            rec[f"mean_{adj}"] = g[adj].mean()
            rec[f"median_{adj}"] = g[adj].median()
            rec[f"share_positive_{adj}"] = float((g[adj] > 0).mean())

        med_abs = g["abs_adj_r_60m"].median()
        lo, hi = st.block_bootstrap_ci(g, "abs_adj_r_60m", "block_d", stat=np.median)
        rec.update({
            "median_abs_adj_r_60m": med_abs,
            "abs_r_60m_ci_low": lo,
            "abs_r_60m_ci_high": hi,
            "mean_abs_adj_r_60m": g["abs_adj_r_60m"].mean(),
            "baseline_abs_r_60m": g["baseline_abs_r_60m"].median(),
        })

        vlo, vhi = st.block_bootstrap_ci(g, "vol_ratio_60m", "block_d", stat=np.median)
        rec.update({
            "median_vol_ratio_5m": g["vol_ratio_5m"].median(),
            "median_vol_ratio_60m": g["vol_ratio_60m"].median(),
            "mean_vol_ratio_60m": g["vol_ratio_60m"].mean(),
            "vol_ratio_60m_ci_low": vlo,
            "vol_ratio_60m_ci_high": vhi,
            "median_vol_ratio_d0": g["vol_ratio_d0"].median(),
            "median_rv_ratio_60m": g["rv_ratio_60m"].median(),
            "median_range_ratio_60m": g["range_ratio_60m"].median(),
        })

        base = rec["baseline_abs_r_60m"]
        cond_size = np.isfinite(base) and base > 0 and med_abs >= config.STRONG_RETURN_MULTIPLE * base
        cond_vol = rec["median_vol_ratio_60m"] >= config.STRONG_VOLUME_RATIO
        cond_evidence = np.isfinite(lo) and np.isfinite(base) and lo > base
        rec.update({
            "return_multiple_of_baseline": med_abs / base if base else np.nan,
            "strong_size": bool(cond_size),
            "strong_volume": bool(cond_vol),
            "strong_evidence": bool(cond_evidence),
            "strong_impact": bool(cond_size and cond_vol and cond_evidence),
            "inference_supported": len(g) >= config.MIN_EVENTS_FOR_INFERENCE,
        })
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("median_abs_adj_r_60m", ascending=False)
    return out.reset_index(drop=True)


def subject_tests(events: pd.DataFrame) -> pd.DataFrame:
    """The three tests of whether subject is related to response at all."""
    df = events[events["exclusion_reason"] == ""].copy()
    df["log_vol_ratio_60m"] = np.log(df["vol_ratio_60m"].replace(0, np.nan))
    df["block_m"] = st.block_label(df, 20)

    rows = []
    for label, col in [("absolute peer-adjusted 60m return", "abs_adj_r_60m"),
                       ("log volume ratio, 60m", "log_vol_ratio_60m"),
                       ("realised volatility ratio, 60m", "rv_ratio_60m")]:
        kw = st.kruskal_by_group(df, "subject", col)
        pm = st.permutation_subject_test(df, "subject", col)
        rows.append({
            "response_measure": label,
            "test": "Kruskal-Wallis across subjects",
            "statistic": kw["H"], "p_value": kw["p"],
            "n": kw["n"], "groups": kw["k"], "effect_size_epsilon_sq": kw["epsilon_sq"],
        })
        rows.append({
            "response_measure": label,
            "test": "permutation, labels shuffled within stock",
            "statistic": pm["observed"], "p_value": pm["p"],
            "n": pm["n"], "groups": kw["k"], "effect_size_epsilon_sq": np.nan,
        })

    try:
        res, ftest = st.clustered_ols(df, "abs_adj_r_60m", "subject", "symbol", "block_m")
        rows.append({
            "response_measure": "absolute peer-adjusted 60m return",
            "test": "OLS, subject dummies + stock FE, SE clustered by stock-month",
            "statistic": float(ftest.fvalue), "p_value": float(ftest.pvalue),
            "n": int(res.nobs), "groups": np.nan,
            "effect_size_epsilon_sq": float(res.rsquared),
        })
    except Exception as exc:  # pragma: no cover - reported, never silently dropped
        rows.append({
            "response_measure": "absolute peer-adjusted 60m return",
            "test": f"OLS failed: {type(exc).__name__}",
            "statistic": np.nan, "p_value": np.nan, "n": np.nan,
            "groups": np.nan, "effect_size_epsilon_sq": np.nan,
        })

    return pd.DataFrame(rows)


def stock_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Impact by stock, and by stock crossed with subject.

    The same announcement type does not move every stock equally, so a
    subject-only table hides where the effect lives. Rows below the inference
    threshold are marked rather than dropped.
    """
    df = events[events["exclusion_reason"] == ""].copy()
    df["block_d"] = st.block_label(df, 1)

    rows = []
    for keys, g in list(df.groupby("symbol")) + list(df.groupby(["symbol", "subject"])):
        sym, subj = (keys, "ALL SUBJECTS") if isinstance(keys, str) else keys
        lo, hi = st.block_bootstrap_ci(g, "abs_adj_r_60m", "block_d", stat=np.median)
        rows.append({
            "symbol": sym,
            "subject": subj,
            "n_events": len(g),
            "median_abs_adj_r_60m": g["abs_adj_r_60m"].median(),
            "mean_abs_adj_r_60m": g["abs_adj_r_60m"].mean(),
            "ci_low": lo,
            "ci_high": hi,
            "median_adj_r_60m": g["adj_r_60m"].median(),
            "share_positive_60m": float((g["adj_r_60m"] > 0).mean()),
            "median_vol_ratio_60m": g["vol_ratio_60m"].median(),
            "median_rv_ratio_60m": g["rv_ratio_60m"].median(),
            "median_adj_r_to_close": g["adj_r_to_close"].median(),
            "inference_supported": len(g) >= config.MIN_EVENTS_FOR_INFERENCE,
        })

    return pd.DataFrame(rows).sort_values(
        ["symbol", "n_events"], ascending=[True, False]).reset_index(drop=True)


def robustness_checks(events: pd.DataFrame) -> pd.DataFrame:
    """Do the subject conclusions survive different treatments of the awkward cases?

    Three specifications, each attacking a different objection:

      all events        the headline specification.
      no overlap        drops the 27% of events with another event inside 60
                        minutes, where the response cannot be attributed to one
                        subject.
      winsorised 1/99   caps the extreme tails. Nothing is deleted for being
                        inconvenient; the check exists to show whether a
                        conclusion rests on a handful of outliers.

    A conclusion that only holds in one column is not a conclusion.
    """
    base = events[events["exclusion_reason"] == ""].copy()

    specs = {"all events": base, "no overlap": base[~base["overlap_60m"]]}

    w = base.copy()
    for col in ["abs_adj_r_60m", "adj_r_60m", "vol_ratio_60m"]:
        lo, hi = w[col].quantile([0.01, 0.99])
        w[col] = w[col].clip(lo, hi)
    specs["winsorised 1/99"] = w

    rows = []
    for subj in sorted(base["subject"].unique()):
        rec = {"subject": subj}
        for name, frame in specs.items():
            g = frame[frame["subject"] == subj]
            rec[f"n [{name}]"] = len(g)
            rec[f"median_abs_r_60m [{name}]"] = g["abs_adj_r_60m"].median()
            rec[f"median_vol_ratio [{name}]"] = g["vol_ratio_60m"].median()
        rows.append(rec)

    out = pd.DataFrame(rows)

    tests = []
    for name, frame in specs.items():
        kw = st.kruskal_by_group(frame, "subject", "abs_adj_r_60m")
        tests.append({"subject": f"__Kruskal-Wallis [{name}]__",
                      f"n [{name}]": kw["n"],
                      f"median_abs_r_60m [{name}]": kw["p"],
                      f"median_vol_ratio [{name}]": kw["epsilon_sq"]})
    return pd.concat([out, pd.DataFrame(tests)], ignore_index=True)


def event_level_table(events: pd.DataFrame, excluded: pd.DataFrame = None) -> pd.DataFrame:
    """The required per-event export, ordered for reading rather than for code.

    Filings that could not be aligned are appended rather than dropped, each
    carrying the reason. An exclusion that never appears in the output is
    indistinguishable from an oversight.
    """
    cols = [
        "cluster_id", "symbol", "NEWSID", "anchor_NEWSID", "event_ts", "ts_source",
        "effective_ts", "pre_bar_ts", "pre_close", "D0", "arrival_bucket", "tradable_regime",
        "delay_minutes", "subject", "subject_rule", "member_subjects", "n_filings",
        "n_companion_filings", "cluster_span_hours", "fiscal_period",
        "overlap_60m", "overlap_same_session",
        "d0_short_session", "truncated_5m", "truncated_30m", "truncated_60m", "incomplete_20d",
        "zero_vol_share_60m",
    ]
    cols += [c for c in RETURN_WINDOWS]
    cols += [f"adj_{c}" for c in RETURN_WINDOWS]
    cols += [f"drift_{h}d" for h in config.DAILY_HORIZONS]
    cols += ["abs_adj_r_60m", "baseline_abs_r_60m",
             "vol_5m", "vol_ratio_5m", "vol_ratio_30m", "vol_ratio_60m", "vol_ratio_d0",
             "rv_60m", "rv_ratio_60m", "range_60m", "range_ratio_60m",
             "exclusion_reason"]

    present = [c for c in cols if c in events.columns]
    out = events[present].copy()

    if excluded is not None and len(excluded):
        drop = excluded[excluded["exclusion_reason"].ne("")].copy()
        if len(drop):
            drop = drop[[c for c in present if c in drop.columns]]
            out = pd.concat([out, drop], ignore_index=True)

    return out.sort_values(["symbol", "effective_ts"], na_position="last").reset_index(drop=True)
