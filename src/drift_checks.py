"""Two checks the drift result has to survive to be worth reporting.

The headline finding -- peer-adjusted drift of about +2.1% by ten sessions --
rests on 57 events and one specification. Two objections can be raised against
it without looking at any new data, and both can be answered with the data in
hand.

**Is it mechanical?** Conditioning on the sign of the day-zero move and then
measuring what follows will produce a positive average whenever the underlying
series has any short-horizon momentum at all, earnings or no earnings. The
placebo test runs the identical construction on random non-earnings sessions
for the same stocks, matched to the same per-stock composition, and asks where
the observed drift sits in that null. If earnings days are not special, the
observed value lands in the middle of the placebo distribution.

**Is the benchmark contaminated?** 72% of earnings events have another of the
five stocks reporting within a fortnight, so the peer basket subtracted from a
stock's return often contains peers inside their own post-earnings window. The
clean-basket check rebuilds the adjustment from only those peers that are not
near an earnings date of their own, and reports how much of the sample still
has a usable basket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats_tools as st
from .pead import PEAD_HORIZONS

#: Sessions either side of an earnings date that a placebo day must avoid, so
#: that a placebo draw is a genuinely ordinary session.
EXCLUSION_RADIUS = 25

#: A peer only contaminates the window it is actually drifting through, so the
#: clean-basket rule is horizon-specific rather than a fixed radius: a peer is
#: dropped for horizon k if it reported within PEER_LOOKBACK sessions before
#: day zero or at any point up to k sessions after it. A flat +/-25 radius
#: leaves 43 of 57 events with no peer at all and answers a harder question
#: than the one being asked.
PEER_LOOKBACK = 5


def daily_close_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Session-end close for every symbol on one shared session index."""
    closes = {}
    for sym, df in bars.items():
        closes[sym] = df.groupby("session_date")["close"].last()
    return pd.DataFrame(closes).sort_index()


def _adjusted_returns(panel: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-session forward return minus the equal-weighted return of the others.

    Matches the peer adjustment used for the headline numbers, so the placebo
    and the observed statistic are computed the same way.
    """
    fwd = panel.shift(-k) / panel - 1.0
    n = panel.shape[1]
    peer_mean = (fwd.sum(axis=1).values[:, None] - fwd.values) / (n - 1)
    return pd.DataFrame(fwd.values - peer_mean, index=panel.index, columns=panel.columns)


def placebo_drift_null(pead_events: pd.DataFrame, bars: dict[str, pd.DataFrame],
                       iters: int = 2000, seed: int = None) -> pd.DataFrame:
    """Where does the observed drift sit among drifts measured on ordinary days?

    Each replication redraws one placebo session per real earnings event, from
    the same stock, keeping the per-stock counts identical so the null has the
    sample's own composition rather than an idealised one.
    """
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    panel = daily_close_panel(bars)
    sessions = panel.index

    # Day-zero move, and the signed drift that follows it, for every session.
    r0 = _adjusted_returns(panel, 1).shift(1)          # close(p-1) -> close(p)
    drift = {k: _adjusted_returns(panel, k) for k in PEAD_HORIZONS}
    sign0 = np.sign(r0)

    # Sessions too close to a real earnings date are not ordinary days.
    eligible: dict[str, np.ndarray] = {}
    for sym, g in pead_events.groupby("symbol"):
        pos = sessions.get_indexer(pd.to_datetime(g["D0"]).dt.normalize())
        blocked = set()
        for p in pos[pos >= 0]:
            blocked.update(range(p - EXCLUSION_RADIUS, p + EXCLUSION_RADIUS + 1))
        ok = [p for p in range(1, len(sessions) - max(PEAD_HORIZONS))
              if p not in blocked and np.isfinite(r0[sym].iloc[p])]
        eligible[sym] = np.array(ok)

    counts = pead_events["symbol"].value_counts().to_dict()
    rows = []

    for k in PEAD_HORIZONS:
        signed = (sign0 * drift[k]).to_numpy()
        col = {s: i for i, s in enumerate(panel.columns)}

        draws = np.empty(iters)
        for it in range(iters):
            vals = []
            for sym, n_ev in counts.items():
                pool = eligible[sym]
                if len(pool) == 0:
                    continue
                pick = rng.choice(pool, size=n_ev, replace=True)
                vals.append(signed[pick, col[sym]])
            v = np.concatenate(vals)
            draws[it] = np.nanmean(v)

        observed = pead_events[f"signed_adj_drift_{k}d"].mean()
        p_two = float(np.mean(np.abs(draws - np.nanmean(draws)) >= abs(observed - np.nanmean(draws))))
        rows.append({
            "horizon_sessions": k,
            "observed_mean_drift": observed,
            "placebo_mean": float(np.nanmean(draws)),
            "placebo_sd": float(np.nanstd(draws, ddof=1)),
            "placebo_p2.5": float(np.nanpercentile(draws, 2.5)),
            "placebo_p97.5": float(np.nanpercentile(draws, 97.5)),
            "outside_placebo_95": bool(
                observed < np.nanpercentile(draws, 2.5) or observed > np.nanpercentile(draws, 97.5)),
            "p_value_vs_placebo": p_two,
            "n_replications": iters,
        })

    return pd.DataFrame(rows)


def clean_peer_drift(pead_events: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Redo the drift using only peers that are not near their own earnings.

    A peer inside its own post-earnings window is exactly the contamination the
    adjustment is supposed to remove, so it is dropped from that event's basket.
    Events left with no clean peer at all are reported rather than imputed.
    """
    panel = daily_close_panel(bars)
    sessions = panel.index
    symbols = list(panel.columns)

    # Each symbol's own earnings positions on the session index.
    own: dict[str, np.ndarray] = {}
    for sym in symbols:
        g = pead_events[pead_events["symbol"] == sym]
        pos = sessions.get_indexer(pd.to_datetime(g["D0"]).dt.normalize()) if len(g) else np.array([])
        own[sym] = pos[pos >= 0]

    def clean_peers(target: str, p: int, k: int) -> list[str]:
        out = []
        for q in symbols:
            if q == target:
                continue
            e = own[q]
            if len(e) and ((e >= p - PEER_LOOKBACK) & (e <= p + k)).any():
                continue
            out.append(q)
        return out

    fwd = {k: (panel.shift(-k) / panel - 1.0) for k in PEAD_HORIZONS}
    r0_raw = panel / panel.shift(1) - 1.0

    records = []
    for ev in pead_events.itertuples(index=False):
        p = sessions.get_indexer([pd.Timestamp(ev.D0).normalize()])[0]
        if p < 1:
            continue
        rec = {"cluster_id": ev.cluster_id, "symbol": ev.symbol, "D0": ev.D0}

        # Day zero is scored against the peers clean over the shortest window.
        base = clean_peers(ev.symbol, p, 0)
        rec["n_clean_peers"] = len(base)
        if base:
            r0 = r0_raw[ev.symbol].iloc[p] - r0_raw[base].iloc[p].mean()
            rec["R0_clean"] = r0
            s = np.sign(r0)
            for k in PEAD_HORIZONS:
                peers = clean_peers(ev.symbol, p, k)
                rec[f"n_clean_peers_{k}d"] = len(peers)
                if peers:
                    d = fwd[k][ev.symbol].iloc[p] - fwd[k][peers].iloc[p].mean()
                    rec[f"signed_clean_drift_{k}d"] = s * d
        records.append(rec)

    ev_clean = pd.DataFrame(records)
    ev_clean["block_m"] = (ev_clean["symbol"].astype(str) + "|"
                           + pd.to_datetime(ev_clean["D0"]).dt.to_period("M").astype(str))

    rows = []
    for k in PEAD_HORIZONS:
        col = f"signed_clean_drift_{k}d"
        sub = ev_clean.dropna(subset=[col])
        summ = st.one_sample_summary(sub[col], block_df=sub, value_col=col, block_col="block_m")
        rows.append({
            "horizon_sessions": k,
            "n_events": summ["n"],
            "mean_drift": summ["mean"],
            "median_drift": summ["median"],
            "continuation_rate": summ.get("share_positive"),
            "ci_low_blocked": summ["ci_low"],
            "ci_high_blocked": summ["ci_high"],
            "p_wilcoxon": summ["p_wilcoxon"],
            "significant_blocked": bool(
                np.isfinite(summ["ci_low"]) and np.isfinite(summ["ci_high"])
                and (summ["ci_low"] > 0 or summ["ci_high"] < 0)),
            "median_clean_peers": float(ev_clean.get(f"n_clean_peers_{k}d", pd.Series([np.nan])).median()),
            "events_dropped_no_clean_peer": int(len(pead_events) - summ["n"]),
        })

    return pd.DataFrame(rows)
