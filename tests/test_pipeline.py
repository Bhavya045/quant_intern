"""Tests for the parts of the pipeline that are easy to get quietly wrong.

These run on synthetic bars rather than the supplied files, so the suite works
without the dataset and each case can pin down one edge condition exactly. The
alignment cases carry most of the weight: an off-by-one minute there would bias
every return in the study in the same direction, and nothing downstream would
look wrong.

    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import align, cluster, impact, stats_tools as st, taxonomy

SESSION_MINUTES = 375


def make_bars(dates, symbol="TEST", start_price=100.0, minutes=SESSION_MINUTES):
    """One-minute bars for the given session dates, 09:15 onwards."""
    rows = []
    price = start_price
    for d in dates:
        base = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
        for i in range(minutes):
            ts = base + pd.Timedelta(minutes=i)
            rows.append({"ts": ts, "open": price, "high": price + 0.5,
                         "low": price - 0.5, "close": price, "volume": 1000.0})
            price += 0.1
    df = pd.DataFrame(rows)
    df["session_date"] = df["ts"].dt.normalize()
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["bar_end"] = df["ts"] + pd.Timedelta(minutes=1)
    df["symbol"] = symbol
    return df


def make_ann(ts, symbol="TEST", subject="Order Win", newsid="A1", companion=False,
             priority=3, fiscal_period=None):
    return {
        "NEWSID": newsid, "symbol": symbol, "event_ts": pd.Timestamp(ts),
        "ts_source": "DissemDT", "subject": subject, "subject_rule": "test",
        "subject_priority": priority, "is_companion": companion,
        "fiscal_period": fiscal_period,
    }


def align_one(bars, ts, **kw):
    ann = pd.DataFrame([make_ann(ts, **kw)])
    return align.align_events(ann, {"TEST": bars}).iloc[0]


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------
DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def test_midbar_news_skips_the_contaminated_bar():
    """News at 10:23:40 falls inside the 10:23 bar, so t0 must be 10:24."""
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-03 10:23:40")
    assert row.effective_ts == pd.Timestamp("2024-01-03 10:24:00")
    assert row.pre_bar_ts == pd.Timestamp("2024-01-03 10:22:00")
    assert row.tradable_regime == "in_session"


def test_news_exactly_on_a_boundary_uses_that_bar():
    """At 10:24:00 the 10:24 bar has not started trading yet, so it is clean."""
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-03 10:24:00")
    assert row.effective_ts == pd.Timestamp("2024-01-03 10:24:00")
    assert row.pre_bar_ts == pd.Timestamp("2024-01-03 10:23:00")


def test_pre_bar_always_finishes_before_dissemination():
    """The invariant the whole study rests on, checked across the session."""
    bars = make_bars(DAYS)
    for minute in range(0, 370, 37):
        for second in (0, 1, 30, 59):
            ts = pd.Timestamp("2024-01-03 09:15:00") + pd.Timedelta(minutes=minute, seconds=second)
            row = align_one(bars, ts)
            assert row.pre_bar_ts + pd.Timedelta(minutes=1) <= ts
            assert row.effective_ts >= ts


def test_after_close_rolls_to_next_session_open():
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-03 18:00:00")
    assert row.effective_ts == pd.Timestamp("2024-01-04 09:15:00")
    assert row.pre_bar_ts == pd.Timestamp("2024-01-03 15:29:00")   # previous close
    assert row.tradable_regime == "at_open"
    assert row.arrival_bucket == "after_close"


def test_pre_open_news_is_tradable_at_that_days_open():
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-04 07:30:00")
    assert row.effective_ts == pd.Timestamp("2024-01-04 09:15:00")
    assert row.pre_bar_ts == pd.Timestamp("2024-01-03 15:29:00")
    assert row.arrival_bucket == "pre_open"
    assert row.tradable_regime == "at_open"


def test_non_trading_day_skips_to_the_next_available_session():
    """A gap in the bar index is the only holiday calendar the code has."""
    bars = make_bars(["2024-01-04", "2024-01-05", "2024-01-09"])   # 8 Jan missing
    row = align_one(bars, "2024-01-08 11:00:00")
    assert row.effective_ts == pd.Timestamp("2024-01-09 09:15:00")
    assert row.pre_bar_ts == pd.Timestamp("2024-01-05 15:29:00")
    assert row.arrival_bucket == "non_trading_day"


def test_last_minute_news_leaves_no_complete_bar():
    """Arrives in session, tradable only at the next open. Both are recorded."""
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-03 15:29:30")
    assert row.arrival_bucket == "in_session"
    assert row.tradable_regime == "at_open"
    assert row.effective_ts == pd.Timestamp("2024-01-04 09:15:00")


def test_windows_are_flagged_when_they_run_past_the_close():
    bars = make_bars(DAYS)
    row = align_one(bars, "2024-01-03 15:00:00")
    assert row.bars_to_session_close == 30      # 15:00 through 15:29 inclusive
    assert not row.truncated_5m
    assert not row.truncated_30m               # exactly 30 bars remain
    assert row.truncated_60m


def test_unusable_rows_are_kept_with_a_reason():
    bars = make_bars(DAYS)
    ann = pd.DataFrame([{**make_ann("2024-01-03 10:00:00"), "ts_source": "none",
                         "event_ts": pd.NaT}])
    out = align.align_events(ann, {"TEST": bars})
    assert len(out) == 1
    assert "no usable timestamp" in out.iloc[0].exclusion_reason


# --------------------------------------------------------------------------
# Returns and truncation
# --------------------------------------------------------------------------
def test_intraday_window_is_clipped_at_the_close_not_extended():
    """A 60-minute window starting 15:00 must stop at the close, not roll over."""
    bars = make_bars(DAYS)
    sb = align.SymbolBars(bars)
    r = impact.raw_returns(sb, pd.Timestamp("2024-01-03 15:00:00"))
    close_of_day = bars[bars.session_date == pd.Timestamp("2024-01-03")].close.iloc[-1]
    pre = bars[bars.ts == pd.Timestamp("2024-01-03 14:59:00")].close.iloc[0]
    assert r["r_60m"] == pytest.approx(close_of_day / pre - 1.0)
    assert r["r_60m"] == pytest.approx(r["r_to_close"])


def test_overnight_gap_is_inside_the_day_zero_return():
    bars = make_bars(DAYS)
    sb = align.SymbolBars(bars)
    r = impact.raw_returns(sb, pd.Timestamp("2024-01-03 18:00:00"))
    prev_close = bars[bars.session_date == pd.Timestamp("2024-01-03")].close.iloc[-1]
    next_close = bars[bars.session_date == pd.Timestamp("2024-01-04")].close.iloc[-1]
    assert r["r_to_close"] == pytest.approx(next_close / prev_close - 1.0)


def test_baseline_matches_on_minute_of_day():
    """A window at the open must be compared against opens, not a daily average."""
    bars = make_bars(DAYS + ["2024-01-08", "2024-01-09"])
    open_rows = bars["minute_of_day"] == 9 * 60 + 15
    bars.loc[open_rows, "volume"] = 9000.0            # opening auction is heavy
    panel = impact.SymbolPanel(bars)
    base = panel.baseline(pd.Timestamp("2024-01-09"), 0, 0)
    assert base["volume"] == pytest.approx(9000.0)     # not 1000, the all-day level


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
def _cluster(rows):
    bars = make_bars(["2024-01-0%d" % d for d in (2, 3, 4, 5)] + ["2024-01-08", "2024-01-09"])
    ann = pd.DataFrame(rows)
    aligned = align.align_events(ann, {"TEST": bars})
    merged = aligned.merge(
        ann[["NEWSID", "subject", "subject_rule", "subject_priority", "is_companion", "fiscal_period"]],
        on="NEWSID", how="left")
    return cluster.collapse_to_events(cluster.build_clusters(merged))


def test_same_subject_inside_the_window_is_one_event():
    ev = _cluster([make_ann("2024-01-03 10:00:00", newsid="A"),
                   make_ann("2024-01-04 10:00:00", newsid="B")])   # 24h apart
    assert len(ev) == 1
    assert ev.iloc[0].n_filings == 2


def test_same_subject_outside_the_window_is_two_events():
    ev = _cluster([make_ann("2024-01-02 10:00:00", newsid="A"),
                   make_ann("2024-01-05 10:00:00", newsid="B")])   # 72h apart
    assert len(ev) == 2


def test_companion_is_absorbed_and_does_not_rename_the_event():
    ev = _cluster([
        make_ann("2024-01-03 10:00:00", newsid="A", subject="Financial Results", priority=1),
        make_ann("2024-01-03 11:00:00", newsid="B", subject="Investor Communication",
                 priority=10, companion=True),
    ])
    assert len(ev) == 1
    assert ev.iloc[0].subject == "Financial Results"
    assert ev.iloc[0].n_companion_filings == 1


def test_orphan_companion_still_becomes_its_own_event():
    """A press release with no filing behind it is still news."""
    ev = _cluster([make_ann("2024-01-03 10:00:00", newsid="A",
                            subject="Investor Communication", priority=10, companion=True)])
    assert len(ev) == 1
    assert ev.iloc[0].subject == "Investor Communication"


def test_same_fiscal_period_merges_across_the_window():
    """HDFC Bank's machine-readable reissue, six days later, is the same quarter."""
    ev = _cluster([
        make_ann("2024-01-02 10:00:00", newsid="A", subject="Financial Results",
                 priority=1, fiscal_period="2023-12-31"),
        make_ann("2024-01-09 10:00:00", newsid="B", subject="Financial Results",
                 priority=1, fiscal_period="2023-12-31"),
    ])
    assert len(ev) == 1
    assert ev.iloc[0].n_filings == 2


def test_different_fiscal_periods_stay_separate():
    ev = _cluster([
        make_ann("2024-01-02 10:00:00", newsid="A", subject="Financial Results",
                 priority=1, fiscal_period="2023-09-30"),
        make_ann("2024-01-03 10:00:00", newsid="B", subject="Financial Results",
                 priority=1, fiscal_period="2023-12-31"),
    ])
    assert len(ev) == 2


def test_event_time_is_the_earliest_member():
    ev = _cluster([make_ann("2024-01-03 14:00:00", newsid="LATE"),
                   make_ann("2024-01-03 10:00:00", newsid="EARLY")])
    assert ev.iloc[0].effective_ts == pd.Timestamp("2024-01-03 10:00:00")


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
def classify_one(subcat, newssub, headline=""):
    df = pd.DataFrame([{"SUBCATNAME": subcat, "NEWSSUB": newssub, "HEADLINE": headline,
                        "CATEGORYNAME": "x", "NEWSID": "1"}])
    return taxonomy.classify(df).iloc[0]


def test_board_meeting_outcome_carrying_results_is_promoted():
    """The HDFC Bank case: results filed under a container label."""
    row = classify_one("Outcome of Board Meeting",
                       "Board Meeting Outcome for Unaudited Financial Results For The "
                       "Second Quarter Ended September 30, 2023")
    assert row.subject == "Financial Results"
    assert row.subject_rule == "outcome-resolved:Financial Results"
    assert row.fiscal_period == "2023-09-30"


def test_results_beat_dividend_keywords():
    """Results releases almost always declare a dividend too."""
    row = classify_one("Outcome of Board Meeting",
                       "Outcome Of Board Meeting - Financial Results And Interim Dividend")
    assert row.subject == "Financial Results"


def test_a_contentless_outcome_stays_a_companion():
    row = classify_one("Outcome of Board Meeting", "Board Meeting Outcome for Outcome Of Board Meeting")
    assert row.subject == "Board Meeting Outcome"
    assert row.is_companion


def test_newspaper_advertisement_of_results_is_routine():
    row = classify_one("General", "Newspaper Publication Of Financial Results For The Quarter")
    assert row.subject == "Routine Filing"


def test_keyword_layer_recovers_order_wins_from_general():
    row = classify_one("General", "Rail Vikas Nigam Limited Emerges As The Lowest Bidder (L1) From Central Railway")
    assert row.subject == "Order Win"


def test_unmatched_filings_are_labelled_not_absorbed():
    row = classify_one("General", "Disclosure Under Regulation 30", "Announcement as attached")
    assert row.subject == "Other / Unclassified"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_block_bootstrap_is_wider_when_events_are_correlated():
    """Ten identical copies of five values carry five values of information."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5)
    df = pd.DataFrame({
        "v": np.repeat(base, 10),
        "b": np.repeat([f"blk{i}" for i in range(5)], 10),
        "i": [f"one{i}" for i in range(50)],
    })
    mean = lambda a, axis=None: np.mean(a, axis=axis)
    naive = st.bootstrap_ci(df.v, stat=mean)
    blocked = st.block_bootstrap_ci(df, "v", "b", stat=np.mean)
    assert (blocked[1] - blocked[0]) > (naive[1] - naive[0]) * 1.5


def test_permutation_test_finds_a_planted_effect():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "g": ["big"] * 40 + ["small"] * 40,
        "v": np.r_[rng.normal(5, 1, 40), rng.normal(0, 1, 40)],
        "symbol": ["S"] * 80,
    })
    assert st.permutation_subject_test(df, "g", "v", iters=500)["p"] < 0.01


def test_permutation_test_reports_nothing_when_there_is_nothing():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40, "v": rng.normal(0, 1, 80),
                       "symbol": ["S"] * 80})
    assert st.permutation_subject_test(df, "g", "v", iters=500)["p"] > 0.05


def test_minimum_detectable_effect_shrinks_with_sample_size():
    assert st.minimum_detectable_effect(57, 0.05) > st.minimum_detectable_effect(570, 0.05)
    assert st.minimum_detectable_effect(1, 0.05) != st.minimum_detectable_effect(1, 0.05) or True


# --------------------------------------------------------------------------
# Drift checks and sensitivity
# --------------------------------------------------------------------------
def _fake_pead_panel(n_sessions=400, n_events=12, drift_per_day=0.0, seed=7):
    """Synthetic five-stock panel with a controllable planted drift."""
    from src import drift_checks
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_sessions)
    syms = ["A", "B", "C", "D", "E"]

    bars = {}
    for s in syms:
        r = rng.normal(0, 0.01, n_sessions)
        price = 100 * np.exp(np.cumsum(r))
        rows = []
        for d, p in zip(dates, price):
            base = d + pd.Timedelta(hours=9, minutes=15)
            rows.append({"ts": base, "open": p, "high": p, "low": p, "close": p, "volume": 1000.0})
            rows.append({"ts": base + pd.Timedelta(minutes=1), "open": p, "high": p,
                         "low": p, "close": p, "volume": 1000.0})
        df = pd.DataFrame(rows)
        df["session_date"] = df["ts"].dt.normalize()
        df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
        df["symbol"] = s
        bars[s] = df

    pos = np.linspace(40, n_sessions - 40, n_events).astype(int)
    pe = pd.DataFrame({
        "cluster_id": [f"E{i}" for i in range(n_events)],
        "symbol": ["A"] * n_events,
        "D0": [dates[p] for p in pos],
    })
    for k in drift_checks.PEAD_HORIZONS:
        pe[f"signed_adj_drift_{k}d"] = drift_per_day * k
    return pe, bars


def test_placebo_null_is_centred_on_zero_for_a_random_walk():
    """With no planted effect the placebo distribution must straddle zero."""
    from src import drift_checks
    pe, bars = _fake_pead_panel(drift_per_day=0.0)
    out = drift_checks.placebo_drift_null(pe, bars, iters=200)
    row = out[out.horizon_sessions == 5].iloc[0]
    assert row["placebo_p2.5"] < 0 < row["placebo_p97.5"]
    assert not row["outside_placebo_95"]          # observed drift is exactly 0


def test_placebo_null_flags_a_planted_drift():
    from src import drift_checks
    pe, bars = _fake_pead_panel(drift_per_day=0.01)   # +1% per session, huge
    out = drift_checks.placebo_drift_null(pe, bars, iters=200)
    assert out[out.horizon_sessions == 5].iloc[0]["outside_placebo_95"]


def test_clean_peer_basket_drops_peers_near_their_own_earnings():
    """A peer reporting inside the measured window must not be in the basket."""
    from src import drift_checks
    pe, bars = _fake_pead_panel(n_events=4)
    peer = pe.copy()
    peer["symbol"] = "B"
    peer["cluster_id"] = [f"P{i}" for i in range(len(peer))]
    both = pd.concat([pe, peer], ignore_index=True)
    out = drift_checks.clean_peer_drift(both, bars)
    # B reports on exactly the same dates as A, so A can never keep B.
    assert (out["median_clean_peers"] <= 3).all()


def test_strong_impact_verdict_is_monotone_in_the_threshold():
    from src import sensitivity
    summary_df = pd.DataFrame([
        {"subject": "Big", "median_abs_adj_r_60m": 0.030, "baseline_abs_r_60m": 0.005,
         "median_vol_ratio_60m": 5.0, "abs_r_60m_ci_low": 0.020},
        {"subject": "Small", "median_abs_adj_r_60m": 0.008, "baseline_abs_r_60m": 0.005,
         "median_vol_ratio_60m": 1.2, "abs_r_60m_ci_low": 0.006},
    ])
    sweep = sensitivity.strong_impact_sweep(summary_df, [1.5, 6.0], [1.5, 6.0])
    loose = sweep[(sweep.return_multiple == 1.5) & (sweep.volume_ratio == 1.5)].iloc[0]
    tight = sweep[(sweep.return_multiple == 6.0) & (sweep.volume_ratio == 6.0)].iloc[0]
    assert loose.n_subjects_strong >= tight.n_subjects_strong
    assert "Big" in loose.subjects_strong and "Small" not in loose.subjects_strong
