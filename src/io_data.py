"""Loading and caching of the supplied market and announcement files.

The raw one-minute files are parsed once and cached as parquet, because the
five files together hold ~1.39 million bars and re-parsing them on every run
makes iteration painful.
"""
from __future__ import annotations

import pandas as pd

from . import config

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(name: str) -> "pd.Series":
    config.CACHE_DIR.mkdir(exist_ok=True)
    return config.CACHE_DIR / f"{name}.parquet"


def load_bars(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Return one-minute bars for a symbol, sorted and de-duplicated.

    Added columns:
      session_date   the calendar date of the trading session
      minute_of_day  minutes since midnight of the bar's opening minute
      bar_end        the instant the bar closes (ts + 1 minute)
    """
    cache = _cache_path(f"bars_{symbol}")
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    path = config.data_dir() / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date"]).rename(columns={"date": "ts"})
    df = df[["ts", *BAR_COLUMNS]].sort_values("ts").reset_index(drop=True)
    df = df.drop_duplicates(subset="ts", keep="first").reset_index(drop=True)

    df["session_date"] = df["ts"].dt.normalize()
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["bar_end"] = df["ts"] + pd.Timedelta(minutes=1)
    df["symbol"] = symbol

    if use_cache:
        df.to_parquet(cache, index=False)
    return df


def load_all_bars(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    return {s: load_bars(s, use_cache) for s in config.SYMBOLS}


def load_announcements() -> pd.DataFrame:
    """Return the announcement table with a single resolved event timestamp.

    DissemDT is the moment the exchange disseminated the filing and is the
    correct clock for an event study. DT_TM is the documented fallback; NEWS_DT
    is used only if both are unusable. The column ts_source records which field
    each row ended up using so the choice stays auditable.
    """
    path = config.data_dir() / "corporate_announcements.jsonl"
    df = pd.read_json(path, lines=True)

    for col in ["DissemDT", "DT_TM", "NEWS_DT", "News_submission_dt"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["symbol"] = df["SCRIP_CD"].map(config.SCRIP_TO_SYMBOL)

    df["event_ts"] = df["DissemDT"]
    df["ts_source"] = "DissemDT"

    fallback = df["event_ts"].isna() & df["DT_TM"].notna()
    df.loc[fallback, "event_ts"] = df.loc[fallback, "DT_TM"]
    df.loc[fallback, "ts_source"] = "DT_TM"

    fallback2 = df["event_ts"].isna() & df["NEWS_DT"].notna()
    df.loc[fallback2, "event_ts"] = df.loc[fallback2, "NEWS_DT"]
    df.loc[fallback2, "ts_source"] = "NEWS_DT"

    df.loc[df["event_ts"].isna(), "ts_source"] = "none"

    for col in ["NEWSSUB", "HEADLINE", "CATEGORYNAME", "SUBCATNAME"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df.sort_values(["symbol", "event_ts"]).reset_index(drop=True)


def build_close_grid(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Closing prices for every symbol on one shared minute index.

    The five files do not carry exactly the same set of minutes (NYKAA is short
    by two bars), so peer comparisons need an explicit common grid. Gaps are
    forward-filled, which is the right convention for a price that simply did
    not print in that minute.
    """
    frames = {s: d.set_index("ts")["close"] for s, d in bars.items()}
    grid = pd.DataFrame(frames).sort_index()
    return grid.ffill()
