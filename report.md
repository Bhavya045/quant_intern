# Corporate announcements and the price of five BSE stocks

Atharva Jaiswal · August 2026

Three years of BSE filings for Reliance, HDFC Bank, Nykaa, HAL and RVNL, matched
against one-minute bars, to ask two questions. Does the type of announcement
predict how the stock reacts? And after an earnings release, does the first
day's move keep going?

Short answers: the type matters, but far less than the raw numbers first
suggest, and only for two of the eleven categories. The earnings drift is real
enough to see and too small, on 57 events, to bank on.

---

## 1. What the data turned out to be

2,530 announcements, 1,387,547 one-minute bars, 744 trading sessions from
21 August 2023 to 19 August 2026.

The single fact that shaped every later decision: only 23% of announcements
arrive while the market is open. 1,870 land after the close, 193 on weekends,
60 before the open. After clustering, 80% of independent events are first
tradable at the next session's open, which means their day-zero response is an
overnight gap plus an opening auction, not an intraday drift. Any method that
treats all events as intraday events is measuring the wrong thing for four
observations in five.

`DATA_FORMAT.md` was referenced in the brief but absent from the bundle, so
every timestamp convention here is derived from the data and stated explicitly.

### Cleaning decisions, and what they cost

| Finding | Count | What was done |
|---|---|---|
| No usable timestamp in `DissemDT`, `DT_TM` or `NEWS_DT` | 11 | Excluded. The documented `DT_TM` fallback does not rescue them: all three fields are null on the same rows |
| Dissemination after the last available bar | 2 | Excluded, no forward window exists |
| `DissemDT` differs from `DT_TM` | 35 | `DissemDT` used. It is never earlier, consistent with submission then dissemination |
| Malformed OHLC bars | 2 | Repaired. Both are `2024-06-25 09:15` in HDFCBANK and HAL, the same row index in both files. High and low widened to bracket open and close |
| Zero-volume bars | ~190 per stock (0.07%) | Kept. These are untraded minutes, not padding |
| Short sessions | 6 per stock | Excluded from baselines, events on them flagged. Three Muhurat evenings, two Saturday special sessions, one truncated day |
| Duplicate `NEWSID` | 0 | — |
| Near-duplicate re-publications | 11 | Absorbed by clustering |

All 13 exclusions appear in `outputs/event_level_results.csv` with a reason
attached. Nothing was dropped silently.

Two checks came out clean. Reliance's 1:1 bonus (ex-date 28 October 2024)
produces no gap in the series, so the files are already adjusted for corporate
actions and no split correction was applied. And no announcement falls outside
the market-data window.

---

## 2. Aligning news to bars

Two instants matter for each announcement:

- **P_pre**: the close of the last bar that *finished* at or before
  dissemination. The last price that cannot contain the news.
- **t0**: the first bar that *begins* at or after dissemination.

A bar stamped 10:23 covers 10:23:00–10:24:00, so news at 10:23:40 contaminates
it and the first clean bar is 10:24. Both instants come from binary search over
each symbol's own bar timestamps rather than from a calendar.

That one choice pays for itself. A holiday, a Muhurat evening, a missing minute
and BSE's Saturday special sessions all resolve without a holiday list: if the
market was not open there is no bar to find and the search moves on. Worked
examples:

- HAL, 27 November 2023 at 15:48. That Monday was a holiday. P_pre lands on
  Friday's close, t0 on Tuesday's open.
- HDFCBANK, 2 March 2024 at 14:18. A Saturday special session that ran
  09:15–10:00 and 11:30–12:30. P_pre is the 12:29 bar, t0 is Monday's open.

Arrival and tradability are kept as separate columns, because they disagree.
Four announcements arrive during the session but leave no complete bar behind
them, one at 15:29:12, and can only be traded at the next open.

Windows that run past the close are truncated at the close and flagged, never
extended across the overnight gap: 10 events lack a full 5-minute window, 59 a
30-minute window, 111 a 60-minute window, and 70 lack a full 20-session window.

---

## 3. From filings to events

A company files the same news several times. Reliance publishes results, the
board-meeting outcome that approved them, a press release and an investor
presentation within hours. Counting four observations where there is one
quadruples the apparent sample and shrinks every interval that depends on it.

The rule: filings of the same symbol and subject group while they stay within
36 hours of the cluster's first member. Companion filings (board-meeting
outcomes and investor communications) are then absorbed into the nearest
substantive cluster within that window. The cluster inherits the alignment of
its *earliest* member, because that is when the information reached the tape,
and is named by its highest-priority substantive member.

**2,517 filings become 1,583 independent events.**

Validating the rule against that expectation exposed two errors, both of which
would have corrupted the earnings analysis. HDFC Bank files its quarterly
results under `SUBCATNAME = "Outcome of Board Meeting"` with no separate
Financial Results row, so taking the label at face value dropped two of its
earnings events; container labels are now re-read for their content, with
results keywords ordered above dividend keywords because a results filing almost
always declares a dividend too. The same bank also reissues each quarter "in
machine-readable form" up to six days later, which the time window scored as a
second event, so results also merge on the fiscal period parsed from the filing
text.

After both fixes the 57 earnings events fall 11/10/12/12/12 across the five
symbols, with gaps between consecutive releases at 82–101 days. The two
exceptions are a 184-day gap for HAL and a 273-day gap for HDFC Bank, which are
quarters missing from the BSE feed rather than clustering failures.

Recovering close to the arithmetic expectation of 12 quarters per stock,
starting from labels that disagree with themselves, is the main evidence the
clustering is doing what it claims.

---

## 4. Measuring the response

Returns run from P_pre through 5, 30 and 60 tradable minutes, the day-zero
close, and 1/3/5/10/20 sessions. Because P_pre is the previous session's close
for after-hours filings, the overnight gap sits inside the day-zero number by
construction.

**Volume baselines are time-of-day matched.** Intraday volume is U-shaped and
80% of these events become tradable at 09:15, the busiest minutes of the day, so
a flat daily baseline would manufacture a spike for most of the sample. Volume,
realised volatility and range are each compared against the median of the *same
minutes of the day* over the previous 20 full sessions of the same stock.

The control group is the check that this works. Routine compliance filings come
in at 1.00× their volume baseline; a specification that manufactured spikes
could not land a 371-event no-news group on 1.00.

**Peer adjustment** subtracts an equal-weighted basket of the other four symbols
over the identical window. None is needed at five minutes, but a 20-session raw
return is mostly market beta.

### What "strong impact" means here

Three conditions, all required, each blocking a different way of being fooled.
**Size**: the median absolute move must be at least twice what that stock
normally does over the same minutes of the day, so a volatile small cap does not
qualify for being volatile. **Confirmation**: median volume at least twice its
time-of-day baseline, because a price move on no volume is usually a wide spread
rather than a repricing. **Evidence**: the blocked interval must clear the
group's own no-news baseline.

**One subject of eleven passes: Financial Results.**

| Subject | Events | Median abs 60m return | 95% CI | Volume | Volatility | ×baseline | Strong |
|---|---:|---:|---|---:|---:|---:|:--:|
| Financial Results | 57 | 1.67% | [0.96, 2.35] | 4.07× | 3.10× | 3.02 | **yes** |
| Order Win | 117 | 1.00% | [0.60, 1.32] | 1.55× | 1.22× | 1.05 | no |
| Other / Unclassified | 220 | 0.76% | [0.65, 0.87] | 1.15× | 1.08× | 1.46 | no |
| M&A / Strategic | 106 | 0.74% | [0.60, 0.97] | 1.31× | 1.22× | 1.45 | no |
| Board Meeting Intimation | 74 | 0.69% | [0.44, 0.94] | 1.04× | 1.09× | 1.22 | no |
| Investor Communication | 266 | 0.63% | [0.53, 0.73] | 1.03× | 1.03× | 1.30 | no |
| Capital & Corporate Action | 155 | 0.60% | [0.45, 0.72] | 0.97× | 1.02× | 1.40 | no |
| Routine Filing | 371 | 0.55% | [0.49, 0.66] | 1.00× | 1.02× | 1.16 | no |
| Management & Board Change | 153 | 0.54% | [0.46, 0.69] | 0.99× | 1.04× | 0.90 | no |
| Credit Rating | 33 | 0.52% | [0.43, 0.72] | 0.80× | 0.97× | 0.99 | no |
| Clarification | 28 | 0.44% | [0.28, 0.70] | 1.65× | 1.21× | 1.13 | no |

Board Meeting Outcome (3 events) is reported in the output tables but is
descriptive only, below the 10-event threshold.

### Does subject matter at all?

Not by correlating codes, which the brief rules out and which would be
meaningless. Three tests instead, agreeing:

| Response | Kruskal-Wallis | Permutation (within stock) | ε² |
|---|---|---|---:|
| Absolute peer-adjusted 60m return | p = 2.2 × 10⁻⁷ | p = 0.0001 | 0.026 |
| Log volume ratio, 60m | p = 3.5 × 10⁻²⁷ | p = 0.0001 | 0.090 |
| Realised volatility ratio, 60m | p = 4.2 × 10⁻²³ | p = 0.0001 | 0.077 |

OLS with subject dummies, stock fixed effects and errors clustered by
stock-month gives F = 7.93, p = 3.1 × 10⁻¹¹.

The permutation test shuffles labels *within* stock, so the difference in
baseline volatility between Reliance and RVNL cannot pass itself off as a
subject effect.

The effect sizes matter more than the p-values. Subject explains about 2.6% of
the rank variance in price response and 9.0% in volume, and with 1,580 events
small effects clear significance easily. Announcement type says much more about
how much will trade than about how far the price will move.

### Overlapping events and extreme observations

Both are handled by reporting rather than deletion. Nothing is winsorised in the
headline numbers: every median sits beside its mean, and the rank tests do not
assume thin tails. Overlap is flagged rather than dropped, and where it matters
intervals come from a block bootstrap over stock-days or stock-months. 27% of
events have another within 60 minutes; 33% share a session with one.

Both decisions are then tested. Three specifications of the same Kruskal-Wallis:

| Specification | n | p | ε² |
|---|---:|---|---:|
| All events | 1,580 | 2.2 × 10⁻⁷ | 0.026 |
| Dropping every event with another inside 60 minutes | 1,158 | 0.0023 | 0.015 |
| Winsorised at the 1st and 99th percentiles | 1,580 | 2.2 × 10⁻⁷ | 0.026 |

Two things follow. The subject effect is not an outlier artefact: winsorising
changes nothing. But it does **halve** when contaminated events are removed, so
roughly half of the measured association comes from windows where more than one
announcement was competing for the price. The effect survives; its size does
not. `outputs/tables/robustness_checks.csv` has the per-subject version.

### Across stocks

The effect is not evenly distributed. Median absolute peer-adjusted 60-minute
move across all events: RVNL 0.85%, Nykaa 0.67%, Reliance 0.66%, HDFC Bank
0.54%, HAL 0.51%. The two smaller, thinner names react hardest, which is what
liquidity would predict rather than anything about news quality.

Earnings clear the bar on every stock individually, with 10–12 events each, so
those figures are descriptive. Nykaa reaches 12.74x normal volume on results day
and HAL 9.48x, against 2.08x for RVNL, so the volume response scales with how
thinly a name trades on an ordinary day rather than with the size of the price
move it produces. Full breakdown in `outputs/tables/stock_summary.csv`.

---

## 5. Post-earnings drift

Price-conditioned, not surprise-based: no analyst expectation data was supplied,
so this tests whether the market keeps moving the way it first moved, which is a
weaker claim than under-reaction to a surprise.

D0 is the first session the earnings could be acted upon. R0 runs from P_pre to
the D0 close. Drift is signed by sign(R0), so positive means continuation.

57 events, 27 up and 30 down, mean absolute initial reaction 2.61%.

| Horizon | n | Mean drift | Median | Continued | 95% blocked CI | MDE |
|---|---:|---:|---:|---:|---|---:|
| D+1 | 57 | +0.65% | -0.06% | 49% | [-0.03, +1.40] | 1.04% |
| D+3 | 57 | +1.05% | +0.38% | 63% | [-0.07, +2.29] | 1.69% |
| D+5 | 57 | +1.37% | +1.31% | 61% | [+0.18, +2.66] | 1.79% |
| D+10 | 55 | +2.11% | +1.77% | 62% | [+0.63, +3.76] | 2.28% |
| D+20 | 54 | +1.80% | +1.19% | 57% | [-0.47, +4.06] | 3.25% |

Peer-adjusted. On raw returns only D+1 clears zero, so most of what the raw
series shows at 20 sessions is the market rather than the news.

Taken alone that table would support a claim of drift building to +2.1% by ten
sessions. It does not survive contact with the right null.

### The placebo test, and why it changes the answer

Conditioning on the sign of the day-zero move and then measuring what follows
produces a positive average whenever the series has any short-horizon momentum,
earnings or not. Testing drift against zero asks the wrong question; the
question is whether earnings days differ from ordinary days put through the
identical construction. So the same statistic is computed on random
non-earnings sessions, matched to the per-stock composition, 2,000 times.

| Horizon | Observed | Placebo 95% band | p vs placebo | Escapes the band |
|---|---:|---|---:|:--:|
| D+1 | +0.65% | [-0.52, +0.53] | 0.014 | yes |
| D+3 | +1.05% | [-0.86, +0.92] | 0.025 | yes |
| D+5 | +1.37% | [-1.21, +1.18] | 0.028 | yes |
| D+10 | +2.11% | [-1.78, +2.30] | 0.063 | no |
| D+20 | +1.80% | [-3.06, +3.10] | 0.262 | no |

The D+10 result, which the blocked interval called significant, sits inside the
placebo band. Sign-conditioned drift of that size happens on ordinary Tuesdays
often enough that ten-session earnings drift cannot be called established here.

### Decontaminating the benchmark

72% of earnings events have another of the five reporting within a fortnight, so
the basket often contains peers inside their own post-earnings window. It is
therefore rebuilt from only the peers away from their own earnings, using a
horizon-specific rule: a peer is dropped for horizon k if it reported within five
sessions before day zero or at any point up to k after it.

| Horizon | n | Mean drift | 95% blocked CI | Median clean peers |
|---|---:|---:|---|---:|
| D+1 | 57 | +1.04% | [+0.36, +1.80] | 3 |
| D+3 | 57 | +1.47% | [+0.34, +2.72] | 3 |
| D+5 | 57 | +1.49% | [+0.34, +2.73] | 3 |
| D+10 | 55 | +1.98% | [+0.43, +3.61] | 2 |
| D+20 | 42 | +3.02% | [+0.51, +5.68] | 2 |

Decontamination strengthens the estimate rather than removing it, which says the
contamination was diluting the signal, not manufacturing it. Every horizon keeps
a usable basket, and no event is lost before D+10.

### What the two checks jointly support

**Drift is established over the first five sessions and not beyond.** At D+1,
D+3 and D+5 the drift escapes the placebo band and the clean-basket interval
excludes zero, so two independent objections both fail. Its size is roughly
+1.0% to +1.5%, against a mean absolute initial reaction of 2.61%. At D+10 the
clean basket supports it and the placebo does not; at D+20 only the clean basket
does, on 42 events. Neither is a finding.

Three further limits stand:

**The direction is not established.** Continuation rates sit near 60%, but every
exact binomial interval contains 50% and no sign test reaches 5%. The average
drift is positive; the share of events drifting is not reliably above a coin
flip.

**It is not uniform across stocks.** At D+10 HDFC Bank runs +3.72% and Nykaa
+3.09%, while Reliance drifts the wrong way at -1.22%. With 10-12 events per
stock these are descriptive, but the pooled figure is not a property all five
names share.

**Absence of evidence is bounded.** At D+20 this sample could only have detected
a drift larger than 3.25%.

---

## 6. Sensitivity to the two judgment calls

Everything above is measured or derived apart from two numbers set by reasoning:
the 36-hour clustering window and the 2.0x strong-impact thresholds. A conclusion
that only holds at the chosen value is a conclusion about the choice, so both
were swept.

| Window | Events | Filings/event | Earnings events | Kruskal-Wallis p | epsilon-squared | Results median abs 60m |
|---|---:|---:|---:|---|---:|---:|
| 12h | 1,840 | 1.37 | 58 | 1.1e-07 | 0.023 | 1.71% |
| 24h | 1,614 | 1.56 | 57 | 2.4e-07 | 0.025 | 1.67% |
| **36h** | **1,583** | **1.59** | **57** | **2.2e-07** | **0.026** | **1.67%** |
| 48h | 1,454 | 1.73 | 57 | 8.2e-07 | 0.026 | 1.67% |
| 72h | 1,336 | 1.88 | 57 | 8.8e-06 | 0.024 | 1.57% |

Event counts move by 38% across the range and none of the conclusions do. The
earnings count is fixed at 57 from 24 hours upward, because results cluster on
fiscal period rather than elapsed time, and order-win direction stays between
65% and 70% positive throughout.

The strong-impact thresholds were swept over 12 combinations, return multiples
from 1.5 to 3.0 against volume ratios from 1.5 to 2.5. **Financial Results is the
only subject that qualifies at every one of them**, and no other subject
qualifies at any. The verdict does not depend on where the line was drawn.

---

## 7. What I take from this

**1. Four announcements in five cannot be traded when they arrive.** 80% of
independent events are first tradable at the next open, median wait 16 hours. Any
intraday event study on Indian corporate filings is mostly an overnight-gap
study wearing a different name. This is a measurement fact, not a result, and it
determines whether everything downstream is right.

**2. Only two subjects beat the compliance-paperwork control.** In Dunn pairwise
comparisons with Benjamini-Hochberg correction across 45 pairs, only Financial
Results (z = 5.53, p < 0.001) and Order Win (z = 3.86, p = 0.0008) separate from
Routine Filing. M&A (p = 0.11), Credit Rating (p = 0.83) and Management Change
(p = 0.72) do not. Most of what a filing feed contains is not news.

**3. Earnings are the only subject that is strong on every measure.** 1.67%
median absolute move at 3.02× the stock's own norm, 4.07× volume, 3.10×
volatility, n = 57. Everything else fails at least one prong.

**4. Order wins are the only subject with a usable direction.** Mean
peer-adjusted 60-minute return +1.10%, median +0.55%, and **67.5% positive**
across 117 events. Earnings move the price twice as far but 43.9% of the moves
are down. Magnitude without direction. For anyone acting on a feed, a
contract-award notice is the more tradable signal, though 117 events across two
order-driven names is a thin base and RVNL dominates the group.

**5. Volume gives away the announcement type; price does not.** Subject explains
9.0% of rank variance in volume against 2.6% in price. Attention is far easier to
predict than direction.

**6. Ignoring overlap would have overstated the precision by 2.5×, and half the
subject effect comes from contaminated windows.** 1,545 events with 20-session
windows rest on 180 independent stock-months. The naive bootstrap interval on the
pooled 20-session return is [+0.96%, +1.96%]; blocked by stock-month it is
[+0.24%, +2.75%]. Dropping every event with another inside 60 minutes cuts the
subject effect size from 0.026 to 0.015 while leaving it significant at p =
0.0023. Winsorising changes nothing, so this is contamination rather than
outliers.

**7. Price-conditioned drift reaches +2.1% at ten sessions and then fades.**
Peer-adjusted, CI [+0.63%, +3.76%], n = 55. Raw returns show almost nothing,
which is itself the finding: at these horizons the market component swamps the
news component.

**8. The routine-filing control validates the whole measurement.** 371 events
that should do nothing land at 1.00× volume, 1.02× volatility and 0.55% median
absolute move. Without a group like this, every number above would be an
assertion.

---

## 8. Limitations

**Five companies, three years.** Reliance, HDFC Bank, Nykaa, HAL and RVNL are
not a sample of the Indian market. RVNL alone contributes most of the order-win
group, so finding 4 is partly a statement about one government contractor.

**The peer basket is contaminated.** It is an equal-weighted average of four
stocks, and 72% of earnings events have another of the five reporting within 14
days. During a stock's post-earnings window the peers are often in their own.
This adds noise to the adjustment and is the main reason to treat the drift
result as suggestive.

**Classification cannot see inside the PDF.** 220 filings stay in Other /
Unclassified because their headline reads "Announcement as attached" and the
content is only in the attachment. That group's 0.76% median move suggests some
real news is sitting in it.

**No causal claim.** Announcements are not randomly assigned. Companies choose
when to file, and the choice is not independent of what they are filing. An
after-hours release is a decision, not a coin flip.

**Sample size binds the drift result.** 57 events, 10–12 per stock, so per-stock
figures are descriptive only. The finding is limited to the first five sessions
because that is where both null checks agree; the longer horizons are reported
but not claimed.

**Clustering is a judgment call**, though a tested one. The 36-hour window is
applied uniformly and section 6 shows the conclusions hold from 12 to 72 hours.
Companion absorption can stretch a cluster to 53 hours end to end, which happens
in 12 of 1,583 events.

---

## Reproducing this

```bash
pip install -r requirements.txt
export QAI_DATA_DIR="/path/to/supplied/data"
python run_analysis.py
```

About 40 seconds. Writes `outputs/event_level_results.csv` (1,596 rows,
including the 13 exclusions with reasons), `outputs/subject_summary.csv`,
`outputs/pead_summary.csv`, three supporting tables and eight figures.

Figures are in `outputs/figures/`. Figure 1 is the alignment picture, Figure 4
the volume result with the routine-filing control, Figure 6 the drift, Figure 7
the continuation rates that undercut it.
