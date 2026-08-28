"""Chart export.

Every figure carries units on the axes and a caption stating what it shows and
what the reader should not over-read. Where a group falls below the threshold
for inference its bar is drawn but marked, so a small sample is visible rather
than implied.
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, stats_tools as st

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c1666b",
    "muted": "#9aa5b1",
    "grid": "#e3e7eb",
    "ok": "#2f6f4e",
}
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.7,
    "figure.autolayout": False,
})

ANALYSED = [
    "Financial Results", "Order Win", "M&A / Strategic", "Credit Rating",
    "Management & Board Change", "Capital & Corporate Action",
    "Board Meeting Intimation", "Investor Communication", "Routine Filing",
]


def _save(fig, name: str, caption: str):
    """Write the figure with its caption below the axes, never across them.

    The caption is wrapped to the figure's own width and the axes are then
    shrunk by exactly the space the wrapped text needs, so captions cannot
    collide with an axis label however long they get.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    width_in, height_in = fig.get_size_inches()
    chars = max(60, int(width_in * 15))
    lines = textwrap.wrap(caption, chars)
    text = "\n".join(lines)

    line_in = 0.155                                  # ~7.2pt line plus leading
    reserve = (len(lines) * line_in + 0.22) / height_in
    fig.tight_layout(rect=(0, reserve, 1, 1))
    # Anchored at the top of the reserved band so the whole block sits inside
    # it; anchoring lower clips the final line off the canvas.
    fig.text(0.012, reserve * 0.94, text, fontsize=7.2, color="#4a5560",
             va="top", ha="left", linespacing=1.45)

    fig.savefig(config.FIGURE_DIR / f"{name}.png", facecolor="white")
    plt.close(fig)
    return config.FIGURE_DIR / f"{name}.png"


def fig_arrival_timing(events: pd.DataFrame):
    ok = events[events["exclusion_reason"] == ""]
    ct = pd.crosstab(ok["arrival_bucket"], ok["tradable_regime"]).reindex(
        ["pre_open", "in_session", "after_close", "non_trading_day"]).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), width_ratios=[1.15, 1])
    ct.plot(kind="barh", stacked=True, ax=ax1,
            color=[PALETTE["accent"], PALETTE["primary"]], width=0.68)
    ax1.set_xlabel("independent events (count)")
    ax1.set_ylabel("")
    ax1.set_title("When announcements arrive vs when they can be traded")
    ax1.legend(title="first tradable", frameon=False, fontsize=8)
    for c in ct.index:
        ax1.text(ct.loc[c].sum() + 8, list(ct.index).index(c), f"{int(ct.loc[c].sum())}",
                 va="center", fontsize=8, color="#4a5560")

    delay = ok.loc[ok["tradable_regime"] == "at_open", "delay_minutes"] / 60.0
    ax2.hist(delay, bins=40, color=PALETTE["primary"], alpha=0.85)
    ax2.set_xlabel("hours from dissemination to first tradable bar")
    ax2.set_ylabel("events")
    ax2.set_title("Waiting time for after-hours filings")
    ax2.axvline(delay.median(), color=PALETTE["accent"], lw=1.4, ls="--")
    ax2.text(delay.median() + 1.5, ax2.get_ylim()[1] * 0.88,
             f"median {delay.median():.0f}h", color=PALETTE["accent"], fontsize=8)

    return _save(fig, "01_event_alignment",
                 "Figure 1. Arrival time against tradability for 1,583 independent events. Four in five become "
                 "tradable only at the next session's open, so their day-zero response is an overnight gap plus an "
                 "opening auction rather than an intraday drift.")


def fig_clustering(events: pd.DataFrame):
    g = events[events["exclusion_reason"] == ""].groupby("subject").agg(
        events=("cluster_id", "size"), filings=("n_filings", "sum")).sort_values("filings")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(g))
    ax.barh(y + 0.2, g["filings"], height=0.38, color=PALETTE["muted"], label="raw filings")
    ax.barh(y - 0.2, g["events"], height=0.38, color=PALETTE["primary"], label="independent events")
    ax.set_yticks(y, g.index)
    ax.set_xlabel("count")
    ax.set_title("Clustering collapses 2,517 filings into 1,583 independent events")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for i, (e, f) in enumerate(zip(g["events"], g["filings"])):
        ax.text(f + 6, i + 0.2, f"{f / e:.2f}x", fontsize=7.5, va="center", color="#4a5560")

    return _save(fig, "02_event_clustering",
                 "Figure 2. Filings per independent event by subject. Earnings and investor communications compress "
                 "hardest (2.2-2.4 filings per event) because a results release arrives with a board-meeting outcome, "
                 "a press release and a presentation attached.")


def fig_subject_return(events: pd.DataFrame):
    df = events[(events["exclusion_reason"] == "") & events["abs_adj_r_60m"].notna()]
    rows = []
    for subj in ANALYSED:
        g = df[df["subject"] == subj]
        if g.empty:
            continue
        lo, hi = st.block_bootstrap_ci(g.assign(_b=st.block_label(g, 1)), "abs_adj_r_60m", "_b", stat=np.median)
        rows.append({"subject": subj, "n": len(g), "median": g["abs_adj_r_60m"].median(),
                     "lo": lo, "hi": hi})
    r = pd.DataFrame(rows).sort_values("median")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(r))
    colors = [PALETTE["accent"] if n < config.MIN_EVENTS_FOR_INFERENCE else PALETTE["primary"] for n in r["n"]]
    ax.barh(y, r["median"] * 100, color=colors, height=0.62)
    ax.errorbar(r["median"] * 100, y, xerr=[(r["median"] - r["lo"]) * 100, (r["hi"] - r["median"]) * 100],
                fmt="none", ecolor="#2b3a45", elinewidth=1.1, capsize=3)
    routine = r.loc[r["subject"] == "Routine Filing", "median"]
    if len(routine):
        ax.axvline(routine.iloc[0] * 100, color=PALETTE["muted"], ls="--", lw=1.2)
        ax.text(routine.iloc[0] * 100 + 0.03, len(r) - 0.6, "routine-filing baseline",
                fontsize=7.5, color="#4a5560")
    ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(r["subject"], r["n"])])
    ax.set_xlabel("median absolute peer-adjusted 60-minute return (%)")
    ax.set_title("Price response in the first hour, by announcement subject")

    return _save(fig, "03_subject_price_impact",
                 "Figure 3. Median absolute peer-adjusted return over the first 60 tradable minutes, with 95% "
                 "block-bootstrap intervals resampled by stock-day. The dashed line is the routine-filing group, "
                 "which is the closest thing this data has to a no-news control.")


def fig_subject_volume(events: pd.DataFrame):
    df = events[(events["exclusion_reason"] == "") & events["vol_ratio_60m"].notna()]
    rows = []
    for subj in ANALYSED:
        g = df[df["subject"] == subj]
        if g.empty:
            continue
        lo, hi = st.block_bootstrap_ci(g.assign(_b=st.block_label(g, 1)), "vol_ratio_60m", "_b", stat=np.median)
        rows.append({"subject": subj, "n": len(g), "median": g["vol_ratio_60m"].median(), "lo": lo, "hi": hi})
    r = pd.DataFrame(rows).sort_values("median")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(r))
    ax.barh(y, r["median"], color=PALETTE["primary"], height=0.62)
    ax.errorbar(r["median"], y, xerr=[r["median"] - r["lo"], r["hi"] - r["median"]],
                fmt="none", ecolor="#2b3a45", elinewidth=1.1, capsize=3)
    ax.axvline(1.0, color=PALETTE["accent"], lw=1.3)
    ax.text(1.02, -0.7, "1.0x = normal volume for this stock at this time of day",
            fontsize=7.5, color=PALETTE["accent"])
    ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(r["subject"], r["n"])])
    ax.set_xlabel("median 60-minute volume / same-minutes baseline (x)")
    ax.set_title("Trading volume in the first hour, relative to a time-of-day baseline")

    return _save(fig, "04_subject_volume",
                 "Figure 4. Volume over the first 60 tradable minutes against the median of the same minutes of the "
                 "day over the previous 20 full sessions. Routine filings land on 1.00x, which is the check that the "
                 "baseline is doing its job rather than manufacturing a spike.")


def fig_response_decay(events: pd.DataFrame):
    df = events[events["exclusion_reason"] == ""]
    show = ["Financial Results", "Order Win", "M&A / Strategic", "Investor Communication", "Routine Filing"]
    windows = [("abs_adj_r_5m", 5), ("abs_adj_r_30m", 30), ("abs_adj_r_60m", 60), ("abs_adj_r_to_close", 375)]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for subj, color in zip(show, [PALETTE["primary"], PALETTE["ok"], "#7b5ea7", PALETTE["muted"], PALETTE["accent"]]):
        g = df[df["subject"] == subj]
        vals = [g[c].median() * 100 for c, _ in windows]
        ax.plot([w for _, w in windows], vals, marker="o", ms=4.5, lw=1.8, color=color,
                label=f"{subj} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_xticks([5, 30, 60, 375], ["5 min", "30 min", "60 min", "to close"])
    ax.set_xlabel("window after the event becomes tradable (log scale)")
    ax.set_ylabel("median absolute peer-adjusted return (%)")
    ax.set_title("How far the response has run by each window")
    ax.legend(frameon=False, fontsize=8)

    return _save(fig, "05_response_decay",
                 "Figure 5. Median absolute peer-adjusted return by window. Most of the earnings response is already "
                 "in the price within the first five tradable minutes; the routine-filing line is flat, as a control "
                 "group should be.")


def fig_pead(summary: pd.DataFrame, placebo: pd.DataFrame = None, clean: pd.DataFrame = None):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    # The placebo band first, so it reads as the reference the lines sit against.
    if placebo is not None:
        pl = placebo.sort_values("horizon_sessions")
        ax.fill_between(pl["horizon_sessions"], pl["placebo_p2.5"] * 100, pl["placebo_p97.5"] * 100,
                        color=PALETTE["muted"], alpha=0.22, lw=0, zorder=1,
                        label="placebo 95% band (ordinary sessions)")

    series = [("raw", PALETTE["muted"], "raw returns", "--")]
    for measure, color, label, ls in series:
        p = summary[(summary["scope"] == "POOLED") & (summary["measure"] == measure)
                    & (summary["horizon_sessions"] > 0)].sort_values("horizon_sessions")
        ax.plot(p["horizon_sessions"], p["mean_drift"] * 100, marker="o", ms=4, lw=1.6,
                color=color, ls=ls, label=label, zorder=3)

    p = summary[(summary["scope"] == "POOLED") & (summary["measure"] == "peer_adjusted")
                & (summary["horizon_sessions"] > 0)].sort_values("horizon_sessions")
    ax.plot(p["horizon_sessions"], p["mean_drift"] * 100, marker="o", ms=5.5, lw=2.1,
            color=PALETTE["primary"], label="peer-adjusted", zorder=4)

    if clean is not None:
        c = clean.sort_values("horizon_sessions")
        ax.errorbar(c["horizon_sessions"] + 0.28, c["mean_drift"] * 100,
                    yerr=[(c["mean_drift"] - c["ci_low_blocked"]) * 100,
                          (c["ci_high_blocked"] - c["mean_drift"]) * 100],
                    fmt="s", ms=4.5, lw=0, elinewidth=1.3, capsize=3,
                    color=PALETTE["ok"], ecolor=PALETTE["ok"], zorder=5,
                    label="clean peer basket, 95% blocked CI")

    ax.axhline(0, color="#2b3a45", lw=1.0, zorder=2)
    ax.set_xticks(sorted(summary.loc[summary["horizon_sessions"] > 0, "horizon_sessions"].unique()))
    ax.set_xlabel("trading sessions after D0")
    ax.set_ylabel("mean signed drift from the D0 close (%)")
    ax.set_title("Price-conditioned post-earnings drift against two null checks")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    return _save(fig, "06_pead_drift",
                 "Figure 6. Drift after the day-zero close, signed by the direction of the day-zero move, 57 events. "
                 "The grey band is the 95% range of the same statistic computed on random non-earnings sessions, "
                 "which is the null that matters: sign-conditioning alone produces drift whenever a series has any "
                 "momentum. Peer-adjusted drift escapes that band at D+1, D+3 and D+5 but not at D+10 or D+20. Green "
                 "squares rebuild the adjustment from only those peers not near their own earnings date.")


def fig_pead_continuation(summary: pd.DataFrame):
    from scipy import stats as sps
    p = summary[(summary["scope"] == "POOLED") & (summary["measure"] == "peer_adjusted")
                & (summary["horizon_sessions"] > 0)].sort_values("horizon_sessions")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(p))
    rate = p["continuation_rate"].to_numpy() * 100
    n = p["n_events"].to_numpy()
    k = np.round(p["continuation_rate"].to_numpy() * n).astype(int)
    lo, hi = [], []
    for ki, ni in zip(k, n):
        ci = sps.binomtest(int(ki), int(ni), 0.5).proportion_ci(0.95)
        lo.append(ci.low * 100)
        hi.append(ci.high * 100)

    ax.bar(x, rate, color=PALETTE["primary"], width=0.6)
    ax.errorbar(x, rate, yerr=[rate - np.array(lo), np.array(hi) - rate],
                fmt="none", ecolor="#2b3a45", elinewidth=1.1, capsize=4)
    ax.axhline(50, color=PALETTE["accent"], lw=1.4, ls="--")
    ax.text(len(p) - 0.45, 50.6, "coin flip", color=PALETTE["accent"], fontsize=8, ha="right")
    ax.set_xticks(x, [f"D+{h}\n(n={ni})" for h, ni in zip(p["horizon_sessions"], n)])
    ax.set_ylim(20, 85)
    ax.set_ylabel("share of events continuing in the D0 direction (%)")
    ax.set_title("Continuation rate with exact binomial intervals")

    return _save(fig, "07_pead_continuation",
                 "Figure 7. Share of earnings events whose peer-adjusted move continued in the day-zero direction. "
                 "Every interval contains 50%, so the direction of the drift is not established even where its "
                 "average size is.")


def fig_stock_subject_heatmap(events: pd.DataFrame):
    df = events[(events["exclusion_reason"] == "") & events["abs_adj_r_60m"].notna()]
    show = [s for s in ANALYSED if (df["subject"] == s).sum() >= config.MIN_EVENTS_FOR_INFERENCE]
    piv = df[df["subject"].isin(show)].pivot_table(
        index="subject", columns="symbol", values="abs_adj_r_60m", aggfunc="median") * 100
    cnt = df[df["subject"].isin(show)].pivot_table(
        index="subject", columns="symbol", values="abs_adj_r_60m", aggfunc="size")
    piv = piv.reindex(show)
    cnt = cnt.reindex(show)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    im = ax.imshow(piv.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, c = piv.iloc[i, j], cnt.iloc[i, j]
            if np.isfinite(v):
                dark = v > np.nanpercentile(piv.to_numpy(), 65)
                ax.text(j, i, f"{v:.2f}\nn={int(c)}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "#22303a")
    ax.set_title("Median absolute peer-adjusted 60-minute return (%), by subject and stock")
    fig.colorbar(im, ax=ax, shrink=0.82, label="%")
    ax.grid(False)

    return _save(fig, "08_stock_subject_heatmap",
                 "Figure 8. The same subject does not move every stock equally. RVNL and NYKAA respond more to order "
                 "flow and results than Reliance or HDFC Bank, which is consistent with their smaller size and "
                 "thinner books rather than with any difference in news quality.")


def build_all(events: pd.DataFrame, pead_summary: pd.DataFrame,
              placebo: pd.DataFrame = None, clean: pd.DataFrame = None) -> list:
    return [
        fig_arrival_timing(events),
        fig_clustering(events),
        fig_subject_return(events),
        fig_subject_volume(events),
        fig_response_decay(events),
        fig_pead(pead_summary, placebo, clean),
        fig_pead_continuation(pead_summary),
        fig_stock_subject_heatmap(events),
    ]
