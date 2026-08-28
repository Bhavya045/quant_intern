"""Mapping announcements onto tradable one-minute bars.

Two instants matter for every announcement:

  P_pre  the close of the last bar that *finished* at or before dissemination.
         This is the last price that cannot contain any of the news.

  t0     the first bar that *begins* at or after dissemination. A bar stamped
         10:23 spans 10:23:00-10:24:00, so news at 10:23:40 contaminates it and
         the first clean bar is 10:24.

Both are found by binary search over the symbol's own bar timestamps rather
than from a calendar. That one choice handles weekends, exchange holidays,
Muhurat sessions, Saturday special sessions and missing minutes without a
hard-coded holiday list: if the market was not open, there is simply no bar
there and the search moves to the next one that exists.

A consequence worth stating: for anything disseminated after the close, the
search lands P_pre on the previous session's final bar and t0 on the next
session's opening bar, so the overnight gap falls inside the measured response
exactly as the brief requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MINUTE = pd.Timedelta(minutes=1)


class SymbolBars:
    """Array views over one symbol's bars, built for repeated binary search."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ts = df["ts"].to_numpy(dtype="datetime64[ns]")
        self.open = df["open"].to_numpy(float)
        self.high = df["high"].to_numpy(float)
        self.low = df["low"].to_numpy(float)
        self.close = df["close"].to_numpy(float)
        self.volume = df["volume"].to_numpy(float)
        self.minute_of_day = df["minute_of_day"].to_numpy(int)
        self.session_date = df["session_date"].to_numpy(dtype="datetime64[ns]")

        sess = df.groupby("session_date").indices
        self.sessions = np.array(sorted(sess.keys()), dtype="datetime64[ns]")
        self.sess_first = np.array([sess[pd.Timestamp(s)][0] for s in self.sessions])
        self.sess_last = np.array([sess[pd.Timestamp(s)][-1] for s in self.sessions])
        self.sess_len = self.sess_last - self.sess_first + 1
        self.sess_pos = {s: i for i, s in enumerate(self.sessions)}
        self.n = len(df)

    def session_position(self, session_date) -> int:
        return self.sess_pos[np.datetime64(session_date, "ns")]


def _arrival_bucket(ts: pd.Timestamp, is_trading_day: bool) -> str:
    if not is_trading_day:
        return "non_trading_day"
    minute = ts.hour * 60 + ts.minute
    if minute < 9 * 60 + 15:
        return "pre_open"
    if minute < 15 * 60 + 30:
        return "in_session"
    return "after_close"


def align_events(ann: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach P_pre, t0 and the trading regime to every announcement.

    Rows that cannot be aligned keep their place in the table and carry an
    exclusion_reason instead of being dropped.
    """
    index = {s: SymbolBars(d) for s, d in bars.items()}
    records: list[dict] = []

    for row in ann.itertuples(index=False):
        rec = {
            "NEWSID": row.NEWSID,
            "symbol": row.symbol,
            "event_ts": row.event_ts,
            "ts_source": row.ts_source,
            "exclusion_reason": "",
        }

        if row.ts_source == "none" or pd.isna(row.event_ts):
            rec["exclusion_reason"] = "no usable timestamp in DissemDT, DT_TM or NEWS_DT"
            records.append(rec)
            continue

        if row.symbol not in index:
            rec["exclusion_reason"] = "scrip code outside the supplied five symbols"
            records.append(rec)
            continue

        sb = index[row.symbol]
        d = pd.Timestamp(row.event_ts)

        # First bar that begins at or after dissemination.
        t0_target = d.ceil("min")
        i0 = int(np.searchsorted(sb.ts, np.datetime64(t0_target, "ns"), side="left"))

        # Last bar that finished at or before dissemination.
        j_pre = int(np.searchsorted(sb.ts, np.datetime64(d - MINUTE, "ns"), side="right")) - 1

        if i0 >= sb.n:
            rec["exclusion_reason"] = "no tradable bar after dissemination within the data window"
            records.append(rec)
            continue
        if j_pre < 0:
            rec["exclusion_reason"] = "no bar precedes dissemination within the data window"
            records.append(rec)
            continue

        d0 = sb.session_date[i0]
        pre_session = sb.session_date[j_pre]
        pos = sb.session_position(d0)

        is_trading_day = np.datetime64(d.normalize(), "ns") in sb.sess_pos
        arrival = _arrival_bucket(d, is_trading_day)
        # Tradability, as distinct from arrival: news at 15:29:30 arrives in
        # session but leaves no complete bar, so it is only actionable at the
        # next open.
        regime = "in_session" if d0 == np.datetime64(d.normalize(), "ns") and arrival == "in_session" else "at_open"

        rec.update(
            {
                "pre_bar_ts": pd.Timestamp(sb.ts[j_pre]),
                "pre_close": sb.close[j_pre],
                "effective_ts": pd.Timestamp(sb.ts[i0]),
                "t0_index": i0,
                "pre_index": j_pre,
                "D0": pd.Timestamp(d0),
                "d0_session_pos": pos,
                "arrival_bucket": arrival,
                "tradable_regime": regime,
                "delay_minutes": (pd.Timestamp(sb.ts[i0]) - d).total_seconds() / 60.0,
                "bars_to_session_close": int(sb.sess_last[pos] - i0 + 1),
                "d0_session_bars": int(sb.sess_len[pos]),
                "d0_short_session": bool(sb.sess_len[pos] < config.FULL_SESSION_BARS),
                "pre_session_date": pd.Timestamp(pre_session),
                "sessions_after_d0": int(len(sb.sessions) - 1 - pos),
            }
        )

        if rec["pre_close"] <= 0:
            rec["exclusion_reason"] = "non-positive pre-announcement price"

        records.append(rec)

    out = pd.DataFrame(records)

    # If every row was excluded the measurement columns never got created.
    for col in ["bars_to_session_close", "sessions_after_d0"]:
        if col not in out.columns:
            out[col] = np.nan

    # Flag, but do not drop, events whose required windows run past the close.
    for w in config.INTRADAY_WINDOWS:
        out[f"truncated_{w}m"] = out["bars_to_session_close"] < w
    out["incomplete_20d"] = out["sessions_after_d0"] < max(config.DAILY_HORIZONS)
    return out
