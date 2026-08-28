"""Run the whole analysis end to end.

    python run_analysis.py

Reads only the supplied files, writes every table and figure under outputs/.
Nothing else in the repository needs to be run first.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (align, cluster, config, drift_checks, figures, impact, io_data,
                 pead, sensitivity, summary, taxonomy)
from src import stats_tools as st

MERGE_COLS = ["NEWSID", "subject", "subject_rule", "subject_priority",
              "is_companion", "fiscal_period", "NEWSSUB", "HEADLINE",
              "CATEGORYNAME", "SUBCATNAME"]


def step(msg: str, t0: float) -> float:
    print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)
    return t0


def main() -> int:
    t0 = time.time()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config.data_dir()}")
    print("\n1. Loading and validating")
    bars = io_data.load_all_bars()
    ann = io_data.load_announcements()
    bar_audit, bars = validate_bars(bars)
    ann_audit = validate_announcements(ann, bars)
    audit = pd.concat([bar_audit, ann_audit], ignore_index=True)
    audit.to_csv(config.TABLE_DIR / "data_quality_audit.csv", index=False)
    step(f"{sum(len(b) for b in bars.values()):,} bars, {len(ann):,} announcements", t0)

    print("\n2. Classifying and aligning")
    ann = taxonomy.classify(ann)
    aligned = align.align_events(ann, bars)
    merged = aligned.merge(ann[MERGE_COLS], on="NEWSID", how="left")
    step(f"{(merged.exclusion_reason == '').sum():,} of {len(merged):,} filings aligned", t0)

    print("\n3. Clustering into independent events")
    events = cluster.collapse_to_events(cluster.build_clusters(merged))
    step(f"{len(events):,} independent events", t0)

    print("\n4. Measuring price and volume response")
    events = impact.measure(events, bars)
    step("returns, volume ratios and volatility ratios attached", t0)

    print("\n5. Subject-level statistics")
    subj = summary.subject_summary(events)
    stocks = summary.stock_summary(events)
    robust = summary.robustness_checks(events)
    tests = summary.subject_tests(events)
    dunn = st.dunn_posthoc(events[events.exclusion_reason == ""], "subject", "abs_adj_r_60m")
    step(f"{len(subj)} subjects, {len(dunn)} pairwise comparisons", t0)

    print("\n6. Post-earnings drift")
    pead_events = pead.build_pead_events(events)
    pead_summary = pead.summarise(pead_events)
    persistence = {m: pead.drift_persistence(pead_summary, m) for m in ("raw", "peer_adjusted")}
    placebo = drift_checks.placebo_drift_null(pead_events, bars)
    clean = drift_checks.clean_peer_drift(pead_events, bars)
    step(f"{len(pead_events)} earnings events, placebo null and clean-basket checks", t0)

    print("\n7. Sensitivity to the two judgment calls")
    window_sweep = sensitivity.clustering_window_sweep(merged, bars)
    threshold_sweep = sensitivity.strong_impact_sweep(subj)
    step(f"{len(window_sweep)} clustering windows, {len(threshold_sweep)} threshold pairs", t0)

    print("\n8. Writing outputs")
    summary.event_level_table(events, excluded=merged).to_csv(
        config.OUTPUT_DIR / "event_level_results.csv", index=False)
    subj.to_csv(config.OUTPUT_DIR / "subject_summary.csv", index=False)
    pead_summary.to_csv(config.OUTPUT_DIR / "pead_summary.csv", index=False)
    tests.to_csv(config.TABLE_DIR / "subject_relationship_tests.csv", index=False)
    stocks.to_csv(config.TABLE_DIR / "stock_summary.csv", index=False)
    robust.to_csv(config.TABLE_DIR / "robustness_checks.csv", index=False)
    dunn.to_csv(config.TABLE_DIR / "dunn_pairwise_comparisons.csv", index=False)
    pead_events.to_csv(config.TABLE_DIR / "pead_event_level.csv", index=False)
    placebo.to_csv(config.TABLE_DIR / "pead_placebo_null.csv", index=False)
    clean.to_csv(config.TABLE_DIR / "pead_clean_peer_basket.csv", index=False)
    window_sweep.to_csv(config.TABLE_DIR / "sensitivity_cluster_window.csv", index=False)
    threshold_sweep.to_csv(config.TABLE_DIR / "sensitivity_strong_thresholds.csv", index=False)
    paths = figures.build_all(events, pead_summary, placebo, clean)
    step(f"3 result tables, 9 supporting tables, {len(paths)} figures", t0)

    print_headlines(events, subj, tests, pead_summary, persistence, robust,
                    placebo, clean, window_sweep, threshold_sweep)
    print(f"\nDone in {time.time() - t0:.1f}s. Outputs under {config.OUTPUT_DIR}")
    return 0


def validate_bars(bars):
    from src import validate
    return validate.audit_bars(bars)


def validate_announcements(ann, bars):
    from src import validate
    return validate.audit_announcements(ann, bars)


def print_headlines(events, subj, tests, pead_summary, persistence, robust=None,
                    placebo=None, clean=None, window_sweep=None, threshold_sweep=None) -> None:
    ok = events[events.exclusion_reason == ""]
    print("\n" + "=" * 78)
    print("HEADLINES")
    print("=" * 78)
    print(f"independent events                {len(ok):,} from {int(ok.n_filings.sum()):,} filings")
    print(f"tradable only at the next open    {(ok.tradable_regime == 'at_open').mean():.0%}")
    print(f"overlapping another event <60min  {ok.overlap_60m.mean():.0%}")

    print("\nsubject impact, first 60 tradable minutes")
    cols = ["subject", "n_events", "median_abs_adj_r_60m", "median_vol_ratio_60m",
            "median_rv_ratio_60m", "return_multiple_of_baseline", "strong_impact"]
    show = subj[cols].copy()
    show["median_abs_adj_r_60m"] = (show.median_abs_adj_r_60m * 100).round(2)
    for c in ["median_vol_ratio_60m", "median_rv_ratio_60m", "return_multiple_of_baseline"]:
        show[c] = show[c].round(2)
    print(show.to_string(index=False))

    print("\nis subject related to response?")
    for r in tests.itertuples():
        if pd.notna(r.p_value):
            print(f"  {r.response_measure:<38} {r.test:<52} p={r.p_value:.3g}")

    print("\nprice-conditioned post-earnings drift, pooled")
    p = pead_summary[(pead_summary.scope == "POOLED") & (pead_summary.horizon_sessions > 0)]
    for measure in ("raw", "peer_adjusted"):
        g = p[p.measure == measure].sort_values("horizon_sessions")
        print(f"  {measure}:")
        for r in g.itertuples():
            flag = "  <- interval excludes zero" if r.significant_blocked else ""
            print(f"    D+{r.horizon_sessions:<3} n={r.n_events:<3} mean={r.mean_drift * 100:+.2f}%"
                  f"  median={r.median_drift * 100:+.2f}%  continued={r.continuation_rate:.0%}"
                  f"  CI=[{r.ci_low_blocked * 100:+.2f}%, {r.ci_high_blocked * 100:+.2f}%]"
                  f"  MDE={r.min_detectable_effect * 100:.2f}%{flag}")
        print(f"    drift visible through D+{persistence[measure]['last_significant_horizon']}")

    if robust is not None:
        print("\nrobustness: does the subject effect survive?")
        k = robust[robust.subject.str.startswith("__Kruskal")]
        for r in k.itertuples(index=False):
            spec = r.subject.strip("_").replace("Kruskal-Wallis ", "")
            vals = [v for v in r[1:] if pd.notna(v)]
            print(f"  {spec:<26} n={int(vals[0]):<5} p={vals[1]:.3g}  eps^2={vals[2]:.3f}")

    if placebo is not None:
        print("\ndrift vs a placebo null on ordinary sessions")
        for r in placebo.itertuples(index=False):
            mark = "  <- outside the placebo 95%" if r.outside_placebo_95 else ""
            print(f"  D+{r.horizon_sessions:<3} observed={r.observed_mean_drift * 100:+.2f}%"
                  f"  placebo 95%=[{getattr(r, '_4') * 100:+.2f}%, {getattr(r, '_5') * 100:+.2f}%]"
                  f"  p={r.p_value_vs_placebo:.3f}{mark}")

    if clean is not None:
        print("\ndrift using only peers away from their own earnings")
        for r in clean.itertuples(index=False):
            mark = "  <- interval excludes zero" if r.significant_blocked else ""
            print(f"  D+{r.horizon_sessions:<3} n={r.n_events:<3} mean={r.mean_drift * 100:+.2f}%"
                  f"  CI=[{r.ci_low_blocked * 100:+.2f}%, {r.ci_high_blocked * 100:+.2f}%]"
                  f"  peers={r.median_clean_peers:.0f}{mark}")

    if window_sweep is not None:
        print("\nsensitivity: clustering window")
        for r in window_sweep.itertuples(index=False):
            mark = "  <- headline" if r.is_headline_setting else ""
            print(f"  {r.cluster_window_hours:>3}h  events={r.n_events:<5} earnings={r.n_earnings_events:<3}"
                  f"  eps^2={r.kw_epsilon_sq:.3f}  results |r60|={r.median_abs_r60_results * 100:.2f}%{mark}")

    if threshold_sweep is not None:
        n = threshold_sweep.n_subjects_strong.nunique()
        uniq = threshold_sweep.subjects_strong.unique()
        print(f"\nsensitivity: strong-impact thresholds over {len(threshold_sweep)} pairs")
        print(f"  distinct verdicts: {n}  ->  {' | '.join(uniq)}")


if __name__ == "__main__":
    raise SystemExit(main())
