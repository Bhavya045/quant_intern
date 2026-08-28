"""Paths and analysis parameters.

Every tunable number used anywhere in the pipeline lives here so that the
assumptions behind a result can be checked in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["RELIANCE", "HDFCBANK", "NYKAA", "HAL", "RVNL"]

# BSE scrip code -> symbol, taken from the supplied metadata.json.
SCRIP_TO_SYMBOL = {
    500325: "RELIANCE",
    500180: "HDFCBANK",
    543384: "NYKAA",
    541154: "HAL",
    542649: "RVNL",
}


def data_dir() -> Path:
    """Resolve the directory holding the supplied datasets.

    Order of precedence:
      1. the QAI_DATA_DIR environment variable,
      2. data/raw/ inside this repository,
      3. '../Quant Analyst Intern/' beside this repository.
    """
    env = os.environ.get("QAI_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"QAI_DATA_DIR is set to {p}, which does not exist.")

    candidates = [REPO_ROOT / "data" / "raw", REPO_ROOT.parent / "Quant Analyst Intern"]
    for c in candidates:
        if (c / "corporate_announcements.jsonl").exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not locate the supplied datasets. Place them in data/raw/ or set "
        "QAI_DATA_DIR. See data/README.md."
    )


OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = REPO_ROOT / ".cache"

# --- Session geometry -------------------------------------------------------
# Bars are stamped at their opening minute; the 15:29 bar covers 15:29:00-15:30:00.
SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
FULL_SESSION_BARS = 375

# --- Event windows ----------------------------------------------------------
INTRADAY_WINDOWS = [5, 30, 60]          # minutes, measured in one-minute bars
DAILY_HORIZONS = [1, 3, 5, 10, 20]      # trading sessions after D0 (3 is used by the drift analysis)

# --- Baselines --------------------------------------------------------------
# Volume, realised volatility and range are all compared against the median of
# the same minute-of-day over this many prior full-length sessions.
BASELINE_SESSIONS = 20

# --- Event clustering -------------------------------------------------------
# Announcements of the same subject for the same scrip within this many hours
# of a cluster anchor are treated as one underlying event.
CLUSTER_WINDOW_HOURS = 36

# --- Statistics -------------------------------------------------------------
BOOTSTRAP_ITERS = 10_000
PERMUTATION_ITERS = 10_000
RANDOM_SEED = 20260828
ALPHA = 0.05
MIN_EVENTS_FOR_INFERENCE = 10   # groups below this are reported as descriptive only

# --- "Strong impact" definition --------------------------------------------
# All three prongs must hold; see report.md for the rationale.
STRONG_RETURN_MULTIPLE = 2.0    # median |return| vs the stock's non-event norm
STRONG_VOLUME_RATIO = 2.0       # median window volume vs its time-of-day baseline
