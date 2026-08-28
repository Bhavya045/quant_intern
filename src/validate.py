"""Data-quality audit.

Everything this module finds is reported, counted and carried forward as a
flag. Nothing is dropped quietly: rows that cannot be used end up in the event
table with an explicit exclusion_reason instead of disappearing.
"""
from __future__ import annotations

import pandas as pd

from . import config


def _row(check: str, scope: str, count, detail: str, action: str) -> dict:
    return {"check": check, "scope": scope, "count": count, "detail": detail, "action": action}


def audit_bars(bars: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Check the one-minute files and repair the malformed bars.

    Returns the audit rows and the repaired bar frames.
    """
    rows: list[dict] = []
    repaired: dict[str, pd.DataFrame] = {}

    for sym, raw in bars.items():
        df = raw.copy()

        rows.append(_row("bar count", sym, len(df), f"{df.session_date.nunique()} sessions", "kept"))

        dup = int(raw["ts"].duplicated().sum())
        rows.append(_row("duplicate timestamps", sym, dup, "identical bar timestamps",
                         "de-duplicated on load" if dup else "none found"))

        nulls = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
        rows.append(_row("null OHLCV", sym, nulls, "any of O/H/L/C/V missing",
                         "excluded from windows" if nulls else "none found"))

        bad = ~(
            (df["low"] <= df[["open", "close"]].min(axis=1))
            & (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["high"] >= df["low"])
        )
        n_bad = int(bad.sum())
        if n_bad:
            detail = ", ".join(str(t) for t in df.loc[bad, "ts"].head(5))
            # High and low are widened to bracket open and close. This touches
            # two bars in 1.39 million and keeps every session contiguous,
            # which matters more than a sub-rupee change to one bar's range.
            df.loc[bad, "high"] = df.loc[bad, ["open", "high", "close"]].max(axis=1)
            df.loc[bad, "low"] = df.loc[bad, ["open", "low", "close"]].min(axis=1)
            rows.append(_row("invalid OHLC", sym, n_bad, detail,
                             "repaired: high/low widened to bracket O/C"))
        else:
            rows.append(_row("invalid OHLC", sym, 0,
                             "low<=min(O,C)<=max(O,C)<=high holds", "none found"))

        zero = int((df["volume"] == 0).sum())
        rows.append(_row("zero-volume bars", sym, zero, f"{100 * zero / len(df):.2f}% of bars",
                         "kept; treated as genuinely untraded minutes"))

        non_pos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        rows.append(_row("non-positive prices", sym, non_pos, "price <= 0",
                         "excluded" if non_pos else "none found"))

        per_session = df.groupby("session_date").size()
        short = per_session[per_session < config.FULL_SESSION_BARS]
        detail = ", ".join(f"{d.date()} ({n} bars)" for d, n in short.items())
        rows.append(_row("short sessions", sym, len(short), detail,
                         "flagged; excluded from baselines, events on them flagged"))

        repaired[sym] = df

    return pd.DataFrame(rows), repaired


def audit_announcements(ann: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(ann)

    rows.append(_row("announcement count", "all", n, "rows in corporate_announcements.jsonl", "kept"))

    dup_id = int(ann["NEWSID"].duplicated().sum())
    rows.append(_row("duplicate NEWSID", "all", dup_id, "exact identifier collisions",
                     "de-duplicated" if dup_id else "none found"))

    # Same scrip, same headline and subject, disseminated within five minutes:
    # the exchange occasionally re-publishes a filing.
    key = ann.assign(_k=ann["symbol"] + "|" + ann["NEWSSUB"] + "|" + ann["HEADLINE"])
    near = 0
    for _, g in key.dropna(subset=["event_ts"]).groupby("_k"):
        if len(g) > 1:
            gaps = g["event_ts"].sort_values().diff().dt.total_seconds()
            near += int((gaps < 300).sum())
    rows.append(_row("near-duplicate filings", "all", near,
                     "same scrip+subject+headline within 5 minutes",
                     "collapsed by event clustering"))

    unmapped = int(ann["symbol"].isna().sum())
    rows.append(_row("unmapped scrip code", "all", unmapped, "SCRIP_CD outside the supplied five",
                     "excluded" if unmapped else "none found"))

    for src, cnt in ann["ts_source"].value_counts().items():
        action = "excluded: no usable timestamp" if src == "none" else f"event clock = {src}"
        rows.append(_row("timestamp source", "all", int(cnt), src, action))

    disagree = int((ann["DissemDT"] != ann["DT_TM"]).sum())
    later = int((ann["DissemDT"] < ann["DT_TM"]).sum())
    rows.append(_row("DissemDT vs DT_TM", "all", disagree,
                     f"differ in {disagree} rows; DissemDT earlier in {later}",
                     "DissemDT preferred (dissemination, not submission)"))

    valid = ann.dropna(subset=["event_ts"])
    lo = min(d["ts"].min() for d in bars.values())
    hi = max(d["ts"].max() for d in bars.values())
    outside = int(((valid["event_ts"] < lo) | (valid["event_ts"] > hi + pd.Timedelta(days=1))).sum())
    rows.append(_row("outside market coverage", "all", outside,
                     f"market data spans {lo.date()} to {hi.date()}",
                     "excluded" if outside else "none found"))

    tod = valid["event_ts"].dt.hour * 60 + valid["event_ts"].dt.minute
    open_m, close_m = 9 * 60 + 15, 15 * 60 + 30
    rows.append(_row("arrival: before open", "all", int((tod < open_m).sum()),
                     "disseminated before 09:15", "tradable at that session's open"))
    rows.append(_row("arrival: in session", "all", int(((tod >= open_m) & (tod < close_m)).sum()),
                     "disseminated 09:15-15:30", "tradable from the next complete bar"))
    rows.append(_row("arrival: after close", "all", int((tod >= close_m).sum()),
                     "disseminated at or after 15:30", "tradable at the next session's open"))
    rows.append(_row("arrival: weekend", "all", int((valid["event_ts"].dt.dayofweek >= 5).sum()),
                     "disseminated on a Saturday or Sunday", "tradable at the next session's open"))

    return pd.DataFrame(rows)
