"""Price and volume response measurement.

Everything is anchored on P_pre, the last price that cannot contain the news.
Measuring from there rather than from the first post-event bar keeps the jump
itself inside the number, and it is what makes the overnight gap part of the
day-zero response for the four filings in five that arrive after the close.

Baselines are the part worth being careful about. Intraday volume is U-shaped:
heavy at the open, thin at midday, heavy into the close. Comparing a 09:15
window against a flat daily average would make every next-open event look like a
volume explosion, and 80% of these events are next-open events. So volume,
realised volatility and range are each compared against the median of the *same
minutes of the day* over the previous BASELINE_SESSIONS full-length sessions of
the same stock.

Peer adjustment uses an equal-weighted basket of the other four symbols over the
identical window. No index was supplied and none is needed for a five-minute
return, but a twenty-session raw return is mostly market beta, and this removes
the bulk of it using only the data in hand. Five cross-sector names make a crude
proxy and the report says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .align import SymbolBars

MIN_OPEN = 9 * 60 + 15          # 09:15 in minutes past midnight
N_SESSION_MINUTES = config.FULL_SESSION_BARS


def _median(x: np.ndarray) -> float:
    """nanmedian that returns NaN for an all-NaN slice instead of warning.

    Events in the first twenty sessions of the sample have no complete baseline
    window, which is expected rather than exceptional.
    """
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


class SymbolPanel:
    """Bars reshaped to (session x minute-of-day) so baselines are one slice.

    Only full-length sessions become rows. The six short sessions per symbol --
    three Muhurat evenings, two Saturday special sessions and one truncated day
    -- are excluded from the baseline grid because their liquidity is not
    comparable. Events that land on them still get returns; their baselines come
    back missing and the event is flagged.
    """

    def __init__(self, df: pd.DataFrame):
        self.bars = SymbolBars(df)

        counts = df.groupby("session_date").size()
        full = counts[counts == N_SESSION_MINUTES].index
        keep = df["session_date"].isin(full)
        panel = df.loc[keep].copy()
        panel["col"] = panel["minute_of_day"] - MIN_OPEN

        self.full_sessions = np.array(sorted(full), dtype="datetime64[ns]")
        self.full_pos = {s: i for i, s in enumerate(self.full_sessions)}
        shape = (len(self.full_sessions), N_SESSION_MINUTES)

        def mat(col: str) -> np.ndarray:
            return panel.pivot(index="session_date", columns="col", values=col).to_numpy(float).reshape(shape)

        self.close = mat("close")
        self.high = mat("high")
        self.low = mat("low")
        self.volume = mat("volume")

        # Previous bar's close, in chronological order across the grid. Row 0
        # column 0 has no predecessor.
        flat = self.close.ravel()
        prev = np.empty_like(flat)
        prev[0] = np.nan
        prev[1:] = flat[:-1]
        self.prev_close = prev.reshape(shape)

        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.log(self.close / self.prev_close)
        self.sq_logret = np.square(logret)

    def prior_rows(self, session_date, n: int) -> np.ndarray:
        """Row indices of the n full sessions immediately before session_date."""
        cut = np.searchsorted(self.full_sessions, np.datetime64(session_date, "ns"), side="left")
        return np.arange(max(0, cut - n), cut)

    def baseline(self, session_date, c0: int, c1: int) -> dict:
        """Median behaviour over the same minutes on the previous sessions.

        c0 and c1 are inclusive column bounds into the session grid.
        """
        rows = self.prior_rows(session_date, config.BASELINE_SESSIONS)
        empty = {"volume": np.nan, "rv": np.nan, "range_pct": np.nan, "abs_ret": np.nan}
        if len(rows) == 0 or c0 < 0 or c1 >= N_SESSION_MINUTES or c1 < c0:
            return empty

        sl = (rows[:, None], np.arange(c0, c1 + 1)[None, :])
        vol = np.nansum(self.volume[sl], axis=1)
        rv = np.sqrt(np.nansum(self.sq_logret[sl], axis=1))
        hi = np.nanmax(self.high[sl], axis=1)
        lo = np.nanmin(self.low[sl], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rng = (hi - lo) / lo
            ret = self.close[rows, c1] / self.prev_close[rows, c0] - 1.0

        return {
            "volume": _median(vol),
            "rv": _median(rv),
            "range_pct": _median(rng),
            "abs_ret": _median(np.abs(ret)),
        }

    def daily_volume_baseline(self, session_date) -> float:
        rows = self.prior_rows(session_date, config.BASELINE_SESSIONS)
        if len(rows) == 0:
            return np.nan
        return float(np.nanmedian(np.nansum(self.volume[rows], axis=1)))


def _window_bounds(sb: SymbolBars, i0: int, k: int) -> tuple[int, bool]:
    """Last bar index of a k-minute window from i0, clipped at the session close."""
    pos = sb.session_position(sb.session_date[i0])
    end = min(i0 + k - 1, int(sb.sess_last[pos]))
    return end, (i0 + k - 1) > int(sb.sess_last[pos])


def raw_returns(sb: SymbolBars, event_ts: pd.Timestamp) -> dict:
    """Returns for one symbol over the windows implied by an event timestamp.

    Recomputed from the timestamp rather than reusing the event's own indices so
    that the identical function serves the event symbol and its four peers.
    """
    ts = np.datetime64(event_ts.ceil("min"), "ns")
    i0 = int(np.searchsorted(sb.ts, ts, side="left"))
    j = int(np.searchsorted(sb.ts, np.datetime64(event_ts - pd.Timedelta(minutes=1), "ns"), side="right")) - 1
    out: dict[str, float] = {}
    if i0 >= sb.n or j < 0:
        return out

    pre = sb.close[j]
    pos = sb.session_position(sb.session_date[i0])

    for k in config.INTRADAY_WINDOWS:
        end, _ = _window_bounds(sb, i0, k)
        out[f"r_{k}m"] = sb.close[end] / pre - 1.0

    out["r_to_close"] = sb.close[int(sb.sess_last[pos])] / pre - 1.0
    d0_close = sb.close[int(sb.sess_last[pos])]

    for h in config.DAILY_HORIZONS:
        p = pos + h
        if p < len(sb.sessions):
            c = sb.close[int(sb.sess_last[p])]
            out[f"r_{h}d"] = c / pre - 1.0
            out[f"drift_{h}d"] = c / d0_close - 1.0
        else:
            out[f"r_{h}d"] = np.nan
            out[f"drift_{h}d"] = np.nan

    return out


def measure(events: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach returns, volume ratios and volatility ratios to every event."""
    panels = {s: SymbolPanel(d) for s, d in bars.items()}
    peers = {s: [q for q in bars if q != s] for s in bars}
    rows = []

    for ev in events.itertuples(index=False):
        rec: dict = {"cluster_id": ev.cluster_id}
        if ev.exclusion_reason or pd.isna(ev.effective_ts):
            rows.append(rec)
            continue

        panel = panels[ev.symbol]
        sb = panel.bars
        i0 = int(ev.t0_index)
        event_ts = pd.Timestamp(ev.event_ts)

        own = raw_returns(sb, event_ts)
        rec.update(own)

        # Equal-weighted basket of the other four names over the same window.
        peer_vals: dict[str, list[float]] = {}
        for q in peers[ev.symbol]:
            for key, val in raw_returns(panels[q].bars, event_ts).items():
                if np.isfinite(val):
                    peer_vals.setdefault(key, []).append(val)
        for key, val in own.items():
            basket = np.mean(peer_vals[key]) if peer_vals.get(key) else np.nan
            rec[f"peer_{key}"] = basket
            rec[f"adj_{key}"] = val - basket

        # --- volume, volatility and range against a time-of-day baseline ----
        in_grid = pd.Timestamp(ev.D0) in panel.full_pos
        for k in config.INTRADAY_WINDOWS:
            end, _ = _window_bounds(sb, i0, k)
            actual = float(np.nansum(sb.volume[i0:end + 1]))
            rec[f"vol_{k}m"] = actual
            base = (
                panel.baseline(ev.D0, int(sb.minute_of_day[i0] - MIN_OPEN), int(sb.minute_of_day[end] - MIN_OPEN))
                if in_grid else {"volume": np.nan, "rv": np.nan, "range_pct": np.nan, "abs_ret": np.nan}
            )
            rec[f"vol_base_{k}m"] = base["volume"]
            rec[f"vol_ratio_{k}m"] = actual / base["volume"] if base["volume"] else np.nan

            if k == 60:
                seg = slice(i0, end + 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lr = np.diff(np.log(sb.close[max(i0 - 1, 0):end + 1]))
                rv = float(np.sqrt(np.nansum(np.square(lr))))
                hi, lo = float(np.nanmax(sb.high[seg])), float(np.nanmin(sb.low[seg]))
                rec["rv_60m"] = rv
                rec["range_60m"] = (hi - lo) / lo if lo else np.nan
                rec["rv_ratio_60m"] = rv / base["rv"] if base["rv"] else np.nan
                rec["range_ratio_60m"] = rec["range_60m"] / base["range_pct"] if base["range_pct"] else np.nan
                rec["baseline_abs_r_60m"] = base["abs_ret"]

            rec[f"zero_vol_share_{k}m"] = float(np.mean(sb.volume[i0:end + 1] == 0))

        pos = sb.session_position(sb.session_date[i0])
        d0_vol = float(np.nansum(sb.volume[int(sb.sess_first[pos]):int(sb.sess_last[pos]) + 1]))
        base_d = panel.daily_volume_baseline(ev.D0) if in_grid else np.nan
        rec["vol_d0"] = d0_vol
        rec["vol_ratio_d0"] = d0_vol / base_d if base_d else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    merged = events.merge(out, on="cluster_id", how="left")

    for k in config.INTRADAY_WINDOWS + ["to_close"]:
        col = f"r_{k}m" if isinstance(k, int) else f"r_{k}"
        if col in merged:
            merged[f"abs_{col}"] = merged[col].abs()
            merged[f"abs_adj_{col}"] = merged[f"adj_{col}"].abs()
    return merged
