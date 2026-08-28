"""Statistical machinery, and the reasoning behind the choices.

Three properties of this sample drive everything here.

Returns are fat-tailed. One Reliance session can dominate a mean, so every
grouped result reports a median beside the mean and the headline tests are
rank-based rather than normal-theory.

Events overlap. A stock files roughly 300 announcements over three years, so
20-session windows for the same stock share most of their trading days. Those
observations are not independent, and a textbook t-statistic on them is too
large -- sometimes by a factor of two or more. Confidence intervals therefore
come from a block bootstrap that resamples whole (stock, period) blocks, so
overlapping events are drawn or dropped together. The naive interval is
reported next to it so the inflation is visible rather than assumed away.

Subject labels are categorical. The brief rules out coding them as numbers and
correlating, which would be meaningless anyway. The relationship is tested
three ways instead: Kruskal-Wallis across all groups, Dunn's pairwise
comparisons with a Benjamini-Hochberg correction, and a permutation test that
shuffles subject labels within each stock so any stock-level differences in
volatility cannot masquerade as a subject effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def describe(x: pd.Series) -> dict:
    """Mean, median and spread for a single group, outliers included."""
    v = pd.Series(x).dropna().to_numpy(float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan,
                "q25": np.nan, "q75": np.nan, "share_positive": np.nan}
    return {
        "n": len(v),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "share_positive": float(np.mean(v > 0)),
    }


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def bootstrap_ci(x, stat=np.median, iters: int = None, alpha: float = None,
                 seed: int = None) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling observations independently."""
    iters = iters or config.BOOTSTRAP_ITERS
    alpha = alpha or config.ALPHA
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    v = pd.Series(x).dropna().to_numpy(float)
    if len(v) < 3:
        return (np.nan, np.nan)
    draws = rng.integers(0, len(v), size=(iters, len(v)))
    vals = stat(v[draws], axis=1)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def block_bootstrap_ci(df: pd.DataFrame, value_col: str, block_col: str,
                       stat=np.median, iters: int = None, alpha: float = None,
                       seed: int = None) -> tuple[float, float]:
    """Percentile interval that resamples whole blocks of correlated events.

    Blocks are drawn with replacement and every event inside a drawn block comes
    with it, so events sharing trading days stay together. This is the interval
    the report quotes wherever windows can overlap.
    """
    iters = iters or config.BOOTSTRAP_ITERS
    alpha = alpha or config.ALPHA
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    d = df[[value_col, block_col]].dropna()
    if d.empty:
        return (np.nan, np.nan)

    d = d.sort_values(block_col, kind="stable")
    flat = d[value_col].to_numpy(float)
    lens = d.groupby(block_col, sort=False).size().to_numpy()
    n = len(lens)
    if n < 3:
        return (np.nan, np.nan)

    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])

    vals = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, n, size=n)
        take = lens[pick]
        # Gather the drawn blocks by index arithmetic rather than by
        # concatenating a list of arrays. Same draw, and fast enough that the
        # per-cell tables can afford a full-size bootstrap.
        ends = np.cumsum(take)
        within = np.arange(ends[-1]) - np.repeat(ends - take, take)
        vals[i] = stat(flat[np.repeat(starts[pick], take) + within])

    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def block_label(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Block identifier sized to the horizon whose overlap it must absorb.

    Intraday windows only collide inside a session, so the stock-day is enough.
    Twenty-session windows overlap for weeks, so the block widens to the
    stock-month.
    """
    d0 = pd.to_datetime(df["D0"])
    if horizon_days <= 1:
        return df["symbol"].astype(str) + "|" + d0.dt.strftime("%Y-%m-%d")
    return df["symbol"].astype(str) + "|" + d0.dt.to_period("M").astype(str)


# --------------------------------------------------------------------------
# Hypothesis tests
# --------------------------------------------------------------------------
def kruskal_by_group(df: pd.DataFrame, group_col: str, value_col: str,
                     min_n: int = None) -> dict:
    """Kruskal-Wallis across subject groups, with epsilon-squared effect size."""
    min_n = min_n or config.MIN_EVENTS_FOR_INFERENCE
    d = df[[group_col, value_col]].dropna()
    groups = [g[value_col].to_numpy(float) for _, g in d.groupby(group_col) if len(g) >= min_n]
    names = [k for k, g in d.groupby(group_col) if len(g) >= min_n]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "k": len(groups), "n": 0,
                "epsilon_sq": np.nan, "groups": names}

    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {
        "H": float(H), "p": float(p), "k": len(groups), "n": int(n),
        # epsilon-squared: the share of rank variance explained by the grouping.
        "epsilon_sq": float((H - len(groups) + 1) / (n - len(groups))),
        "groups": names,
    }


def dunn_posthoc(df: pd.DataFrame, group_col: str, value_col: str,
                 min_n: int = None) -> pd.DataFrame:
    """Dunn's pairwise rank comparisons with Benjamini-Hochberg correction.

    Implemented directly: with ~10 groups there are 45 pairs, and reporting raw
    p-values across 45 comparisons would guarantee false positives.
    """
    min_n = min_n or config.MIN_EVENTS_FOR_INFERENCE
    d = df[[group_col, value_col]].dropna()
    sizes = d.groupby(group_col).size()
    keep = sizes[sizes >= min_n].index
    d = d[d[group_col].isin(keep)]
    if d[group_col].nunique() < 2:
        return pd.DataFrame()

    d = d.assign(_rank=stats.rankdata(d[value_col].to_numpy(float)))
    N = len(d)
    mean_rank = d.groupby(group_col)["_rank"].mean()
    n_by = d.groupby(group_col).size()

    # Tie correction for the shared variance term.
    _, counts = np.unique(d[value_col].to_numpy(float), return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))

    rows = []
    names = list(mean_rank.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            se = np.sqrt(sigma2 * (1.0 / n_by[a] + 1.0 / n_by[b]))
            z = (mean_rank[a] - mean_rank[b]) / se
            rows.append({"group_a": a, "group_b": b, "n_a": int(n_by[a]), "n_b": int(n_by[b]),
                         "z": float(z), "p_raw": float(2 * stats.norm.sf(abs(z)))})

    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p_raw"], method="fdr_bh")[1]
    out["significant"] = out["p_bh"] < config.ALPHA
    return out.sort_values("p_bh").reset_index(drop=True)


def permutation_subject_test(df: pd.DataFrame, group_col: str, value_col: str,
                             strata_col: str = "symbol", iters: int = None,
                             min_n: int = None, seed: int = None) -> dict:
    """Does subject explain response, once each stock is its own control?

    Labels are shuffled *within* stock. Reliance and RVNL differ enormously in
    baseline volatility, and an unstratified shuffle would let that difference
    leak into the null and overstate the subject effect.

    The statistic is the spread of group median responses.
    """
    iters = iters or config.PERMUTATION_ITERS
    min_n = min_n or config.MIN_EVENTS_FOR_INFERENCE
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    d = df[[group_col, value_col, strata_col]].dropna()
    sizes = d.groupby(group_col).size()
    d = d[d[group_col].isin(sizes[sizes >= min_n].index)]
    if d[group_col].nunique() < 2:
        return {"observed": np.nan, "p": np.nan, "iters": 0, "n": len(d)}

    # Integer codes rather than pandas groupby: this runs 10,000 times, and a
    # groupby per iteration dominated the whole pipeline's runtime.
    codes = pd.Categorical(d[group_col]).codes.astype(np.int64)
    n_groups = codes.max() + 1
    values = d[value_col].to_numpy(float)
    strata = d[strata_col].to_numpy()

    def spread(lab: np.ndarray) -> float:
        meds = np.empty(n_groups)
        for g in range(n_groups):
            meds[g] = np.median(values[lab == g])
        return float(meds.max() - meds.min())

    observed = spread(codes)
    idx_by_stratum = [np.flatnonzero(strata == s) for s in np.unique(strata)]

    count = 0
    shuffled = codes.copy()
    for _ in range(iters):
        for idx in idx_by_stratum:
            shuffled[idx] = rng.permutation(codes[idx])
        if spread(shuffled) >= observed:
            count += 1

    return {"observed": observed, "p": (count + 1) / (iters + 1), "iters": iters, "n": len(d)}


def clustered_ols(df: pd.DataFrame, value_col: str, group_col: str,
                  fe_col: str = "symbol", cluster_col: str = None):
    """Subject dummies with stock fixed effects and cluster-robust errors.

    A parametric cross-check on the rank tests: the joint F-test on the subject
    dummies answers the same question under different assumptions, and standard
    errors clustered by stock-month absorb the overlap the rank tests cannot.
    """
    import statsmodels.formula.api as smf

    d = df[[value_col, group_col, fe_col]].dropna().copy()
    d["_cluster"] = df.loc[d.index, cluster_col] if cluster_col else d[fe_col]
    d = d.rename(columns={value_col: "y", group_col: "subject_g", fe_col: "fe"})

    model = smf.ols("y ~ C(subject_g) + C(fe)", data=d)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["_cluster"]})

    terms = [t for t in res.params.index if t.startswith("C(subject_g)")]
    f_test = res.f_test(" , ".join(f"{t} = 0" for t in terms)) if terms else None
    return res, f_test


# --------------------------------------------------------------------------
# One-sample tests, used by the drift analysis
# --------------------------------------------------------------------------
def one_sample_summary(x, block_df: pd.DataFrame = None, value_col: str = None,
                       block_col: str = None) -> dict:
    """Everything needed to judge whether a drift estimate differs from zero."""
    v = pd.Series(x).dropna().to_numpy(float)
    out = describe(v)
    if len(v) < 3:
        out.update({"t_stat": np.nan, "p_ttest": np.nan, "p_wilcoxon": np.nan,
                    "p_sign": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "ci_low_naive": np.nan, "ci_high_naive": np.nan})
        return out

    t, p_t = stats.ttest_1samp(v, 0.0)
    try:
        _, p_w = stats.wilcoxon(v)
    except ValueError:
        p_w = np.nan
    n_pos = int(np.sum(v > 0))
    p_sign = stats.binomtest(n_pos, len(v), 0.5).pvalue

    lo_n, hi_n = bootstrap_ci(v, stat=lambda a, axis=None: np.mean(a, axis=axis))
    if block_df is not None and value_col and block_col:
        lo, hi = block_bootstrap_ci(block_df, value_col, block_col, stat=np.mean)
    else:
        lo, hi = lo_n, hi_n

    out.update({"t_stat": float(t), "p_ttest": float(p_t), "p_wilcoxon": float(p_w),
                "p_sign": float(p_sign), "n_positive": n_pos,
                "ci_low": lo, "ci_high": hi, "ci_low_naive": lo_n, "ci_high_naive": hi_n})
    return out


def minimum_detectable_effect(n: int, sd: float, alpha: float = None,
                              power: float = 0.80) -> float:
    """Smallest true mean this sample could reliably detect.

    Reported alongside the drift results because "we found nothing" and "we
    could not have found anything smaller than this" are different claims, and
    with 57 earnings events only the second one is defensible.
    """
    alpha = alpha or config.ALPHA
    if not n or n < 2 or not np.isfinite(sd):
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(n))
