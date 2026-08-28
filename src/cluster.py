"""Grouping announcements into independent events.

A company routinely files the same news four times. Reliance publishes its
quarterly results, the outcome of the board meeting that approved them, a press
release and an investor presentation, all inside a couple of hours. Counting
those as four observations would quadruple the apparent sample and make every
confidence interval too narrow.

The rule, applied identically to every symbol and subject:

  1. Filings of the same symbol and the same subject are grouped while they
     stay within CLUSTER_WINDOW_HOURS of the cluster's *anchor* -- the first
     filing in the group. Measuring against the anchor rather than the previous
     member stops a long chain of filings 35 hours apart from collapsing into
     one cluster spanning weeks.

  2. Companion filings -- board-meeting outcomes and investor communications --
     are then absorbed into the nearest substantive cluster of the same symbol
     within the same window. A companion with no substantive cluster nearby
     falls back to rule 1 and forms its own event, because a press release with
     no filing behind it is still news.

  3. The cluster inherits the alignment of its *earliest* member. That is the
     moment the information first reached the tape, which is the moment the
     market could first respond.

  4. The cluster is named by its highest-priority substantive member, so a
     results cluster is a results event even though it also contains a
     presentation and a press release.

Overlap flags are added afterwards so that events competing for the same price
move can be excluded in a robustness check rather than silently averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .taxonomy import SUBJECTS


def build_clusters(events: pd.DataFrame) -> pd.DataFrame:
    """Assign cluster_id to every alignable event.

    Rows carrying an exclusion_reason keep their place and get no cluster.
    """
    df = events.copy()
    df["cluster_id"] = pd.NA

    usable = df["exclusion_reason"].eq("") & df["effective_ts"].notna()
    window = pd.Timedelta(hours=config.CLUSTER_WINDOW_HOURS)
    counter = 0

    # --- rule 1: anchor-window grouping within (symbol, subject) --------------
    def anchor_groups(frame: pd.DataFrame) -> dict:
        nonlocal counter
        assigned = {}
        for (sym, subj), g in frame.groupby(["symbol", "subject"], sort=True):
            anchor = None
            cid = None
            for idx, ts in g.sort_values("effective_ts")["effective_ts"].items():
                if anchor is None or ts - anchor > window:
                    counter += 1
                    cid = f"{sym}-{counter:04d}"
                    anchor = ts
                assigned[idx] = cid
        return assigned

    subs = df[usable & ~df["is_companion"]]
    assigned = anchor_groups(subs)
    assigned = _merge_by_fiscal_period(df, assigned)

    # Anchor time of each substantive cluster, for companion attachment.
    anchor_ts, anchor_sym = {}, {}
    for idx, cid in assigned.items():
        ts = df.at[idx, "effective_ts"]
        if cid not in anchor_ts or ts < anchor_ts[cid]:
            anchor_ts[cid] = ts
        anchor_sym[cid] = df.at[idx, "symbol"]

    # --- rule 2: absorb companions into the nearest substantive cluster ------
    by_symbol: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for cid, ts in anchor_ts.items():
        by_symbol.setdefault(anchor_sym[cid], []).append((ts, cid))
    for sym in by_symbol:
        by_symbol[sym].sort()

    comps = df[usable & df["is_companion"]]
    orphans = []
    for idx, row in comps.iterrows():
        candidates = by_symbol.get(row["symbol"], [])
        best, best_gap = None, window
        for ts, cid in candidates:
            gap = abs(row["effective_ts"] - ts)
            if gap <= best_gap:
                best, best_gap = cid, gap
        if best is not None:
            assigned[idx] = best
        else:
            orphans.append(idx)

    if orphans:
        assigned.update(anchor_groups(df.loc[orphans]))

    df.loc[list(assigned.keys()), "cluster_id"] = pd.Series(assigned)
    return df


def _merge_by_fiscal_period(df: pd.DataFrame, assigned: dict) -> dict:
    """Re-key results clusters on the fiscal period they report on.

    The time window alone treats HDFC Bank's "in machine-readable form" reissue,
    published up to six days after the original, as a second earnings event. The
    period named in the filing text says otherwise.

    The period both merges and separates. Filings naming the same period join
    even when they fall outside the window; filings naming different periods
    stay apart even when they fall inside it.

    It runs on top of the window rule rather than replacing it, because the
    period parses out of only 64 of 93 results filings. A filing with no
    parseable period keeps its window cluster, and follows the earliest period
    present there if that cluster has one, which is what keeps a results release
    together with the same-day filings whose text does not name the quarter.
    """
    if "fiscal_period" not in df.columns:
        return assigned

    period = df["fiscal_period"]

    def period_of(i):
        p = period.get(i)
        return p if isinstance(p, str) and p else None

    # Every (symbol, period) pair becomes one cluster, keyed by the period
    # itself rather than by a window cluster id. Reusing a window id here would
    # fail to separate two periods that happened to land in the same window.
    periods_in_cluster: dict[str, set[str]] = {}
    seen: set[tuple] = set()
    for i, cid in assigned.items():
        p = period_of(i)
        if p is None:
            continue
        seen.add((df.at[i, "symbol"], p))
        periods_in_cluster.setdefault(cid, set()).add(p)

    if not seen:
        return assigned

    canonical = {(sym, p): f"{sym}-FP-{p}" for sym, p in seen}

    out = {}
    for i, cid in assigned.items():
        p = period_of(i)
        if p is None:
            # No period of its own: follow the earliest period in its cluster.
            present = periods_in_cluster.get(cid)
            p = min(present) if present else None
        out[i] = canonical.get((df.at[i, "symbol"], p), cid) if p else cid
    return out


def collapse_to_events(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce clustered announcements to one row per independent event."""
    clustered = df[df["cluster_id"].notna()].copy()
    clustered["_rank"] = list(zip(clustered["subject_priority"], clustered["effective_ts"]))

    rows = []
    for cid, g in clustered.groupby("cluster_id", sort=True):
        g = g.sort_values("effective_ts")
        first = g.iloc[0]                                    # earliest member drives alignment
        namer = g.sort_values(["subject_priority", "effective_ts"]).iloc[0]

        rec = first.to_dict()
        rec.update(
            {
                "cluster_id": cid,
                "subject": namer["subject"],
                "subject_priority": namer["subject_priority"],
                "subject_rule": namer["subject_rule"],
                "anchor_NEWSID": namer["NEWSID"],
                "n_filings": len(g),
                "n_companion_filings": int(g["is_companion"].sum()),
                "member_subjects": " | ".join(sorted(set(g["subject"]))),
                "cluster_span_hours": (g["effective_ts"].max() - g["effective_ts"].min()).total_seconds() / 3600.0,
            }
        )
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["symbol", "effective_ts"]).reset_index(drop=True)
    return _add_overlap_flags(out)


def _add_overlap_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Mark events whose price response is contested by another event.

    Two flags, because they answer different questions: overlap_60m says the
    intraday window is contaminated, overlap_same_session says the day-level
    response cannot be attributed to one subject.
    """
    ev = ev.copy()
    ev["overlap_60m"] = False
    ev["overlap_same_session"] = False

    for sym, g in ev.groupby("symbol"):
        ts = g["effective_ts"].to_numpy(dtype="datetime64[ns]")
        order = np.argsort(ts)
        ts_sorted = ts[order]
        idx = g.index.to_numpy()[order]

        hour = np.timedelta64(60, "m")
        lo = np.searchsorted(ts_sorted, ts_sorted - hour, side="left")
        hi = np.searchsorted(ts_sorted, ts_sorted + hour, side="right")
        ev.loc[idx, "overlap_60m"] = (hi - lo) > 1

        same_day = g["D0"].value_counts()
        ev.loc[g.index, "overlap_same_session"] = g["D0"].map(same_day).gt(1).to_numpy()

    return ev
