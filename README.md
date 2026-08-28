
````markdown
# Corporate announcement impact on five BSE stocks

Event study over **2,530 BSE corporate announcements** and **1.39 million one-minute bars** for Reliance, HDFC Bank, Nykaa, HAL and RVNL, covering **21 August 2023 to 19 August 2026**.

**Full write-up:** [report.md](report.md)

---

## Reproducing

```bash
pip install -r requirements.txt

# Point the pipeline at the supplied files (they are not in this repo)

# Bash
export QAI_DATA_DIR="/path/to/Quant Analyst Intern"

# PowerShell
$env:QAI_DATA_DIR = "C:\path\to\Quant Analyst Intern"

python run_analysis.py
````

One command, about 60 seconds, no manual steps. **Python 3.11.**

The test suite runs on synthetic bars and needs no data:

```bash
pytest -q
```

**33 tests, ~2 seconds.**

Alternatively, copy the files into `data/raw/` and run:

```bash
python run_analysis.py
```

with no environment variable.

See [data/README.md](data/README.md) for the file list.

> **Note:** The supplied datasets are not committed to this repository and are excluded in `.gitignore`.

---

## Outputs

| Path                              | Contents                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs/event_level_results.csv` | 1,596 rows: 1,583 independent events plus 13 exclusions with reasons                                                                     |
| `outputs/subject_summary.csv`     | Impact statistics per subject with bootstrap intervals                                                                                   |
| `outputs/pead_summary.csv`        | Drift by horizon, pooled and per stock, raw and peer-adjusted                                                                            |
| `outputs/tables/`                 | Quality audit, subject tests, Dunn pairwise, stock breakdown, robustness, drift placebo and clean-basket checks, both sensitivity sweeps |
| `outputs/figures/`                | Eight captioned PNGs                                                                                                                     |

---

## Method in brief

### Alignment

`DissemDT` is the event clock, `DT_TM` the fallback.

`P_pre` is the close of the last bar finishing at or before dissemination; `t0` is the first bar beginning at or after it, so a bar straddling the news is never used.

Both come from binary search over each symbol's own bar timestamps, which handles holidays, Muhurat evenings and BSE's split Saturday sessions without a hard-coded calendar.

For the **80% of events first tradable at the next open**, `P_pre` is the previous session's close, putting the overnight gap inside the day-zero response.

### Clustering

Filings of the same symbol and subject group within **36 hours** of the cluster anchor.

Board-meeting outcomes and investor communications are absorbed into the nearest substantive cluster.

Results also merge on the fiscal period parsed from the filing text.

**2,517 filings become 1,583 independent events; the 57 earnings events fall 11/10/12/12/12 across the five stocks.**

### Taxonomy

BSE's `SUBCATNAME` where it exists, then keyword rules over `NEWSSUB` and `HEADLINE` for the **688 filings** BSE files under "General".

Eleven analysed subjects plus an explicit **Other / Unclassified** group.

### Baselines

Volume, realised volatility and range are compared against the median of the **same minutes of the day** over the previous 20 full sessions of the same stock.

Returns are also reported peer-adjusted against an equal-weighted basket of the other four symbols.

### Statistics

Median beside mean throughout.

Kruskal-Wallis, Dunn pairwise with Benjamini-Hochberg, a permutation test shuffling labels within stock, and OLS with stock fixed effects and clustered errors.

Confidence intervals come from a block bootstrap resampling stock-days for intraday windows and stock-months for the 20-session horizon.

### Null checks on the drift

Sign-conditioning produces drift on any series with momentum, so the drift is also tested against a placebo null built from random non-earnings sessions.

The peer basket is rebuilt from only those peers away from their own earnings dates.

### Sensitivity

The clustering window and the strong-impact thresholds are the only numbers set by judgment, so both are swept and the results reported.

---

## Main findings

1. **80% of events cannot be traded when they arrive.** Median wait 16 hours. Most of this data is an overnight-gap study, not an intraday one.

2. **Only two subjects beat the routine-filing control** after correcting for 45 comparisons: Financial Results and Order Win. M&A, Credit Rating and Management Change do not.

3. **Earnings are the only subject strong on every measure:** 1.67% median absolute 60-minute move at 3.02× the stock's own norm, 4.07× volume, `n = 57`.

4. **Order wins are the only subject with a usable direction:** +1.10% mean, 67.5% positive across 117 events.

5. **Volume gives away the announcement type; price does not.** Subject explains 9.0% of rank variance in volume against 2.6% in price.

6. **Overlap inflates naive precision by 2.5×.** 1,545 events with 20-session windows rest on 180 independent stock-months.

7. **Price-conditioned drift is real for about a week.** Over D+1 to D+5 the +1.0% to +1.5% drift escapes a placebo band built from ordinary sessions (`p = 0.014` to `0.028`) and survives decontaminating the peer basket. The +2.1% at D+10 sits inside the placebo band, so it is suggestive, not established.

8. **None of the conclusions depend on the two numbers I chose.** Clustering windows from 12 to 72 hours move the event count 38% and change nothing; Financial Results is the only strong subject at all 12 threshold pairs.

---

## Main limitations

Five companies over three years, so nothing here generalises to the Indian market.

The peer basket is contaminated: **72% of earnings events have another of the five reporting within 14 days.**

**220 filings** stay unclassified because their headline is `"Announcement as attached"` and the content is in a PDF.

The drift result rests on **57 events**, holds only over the first five sessions, and Reliance drifts the opposite way.

No causal claim is made: companies choose when to file.

`DATA_FORMAT.md` was referenced in the brief but not supplied, so all timestamp conventions were derived from the data and are stated explicitly in [report.md](report.md).

---

## Layout

```text
run_analysis.py          single entry point

src/
  config.py              paths and every tunable parameter
  io_data.py             loading, parquet cache
  validate.py            data-quality audit
  align.py               P_pre / t0 / trading regime
  taxonomy.py             subject classification rules
  cluster.py             independent-event construction
  impact.py              returns, volume and volatility with baselines
  stats_tools.py         bootstrap, rank tests, permutation, clustered OLS
  pead.py                price-conditioned drift
  drift_checks.py        placebo null and clean-peer-basket checks
  sensitivity.py         sweeps for the clustering window and thresholds
  summary.py             subject aggregation, strong-impact rule
  figures.py             chart export

tests/
  test_pipeline.py       33 tests on synthetic bars, no data required

outputs/                 tables and figures (committed)

data/raw/                supplied files (gitignored)
```

````


Then GitHub will render it properly with headings, tables, code blocks, and bullets.
