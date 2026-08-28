"""Rule-based announcement subject taxonomy.

Two layers, applied in order:

  1. An exact match on BSE's own SUBCATNAME. Where the exchange has already
     classified a filing, that label is trusted.
  2. A keyword pass over NEWSSUB and HEADLINE, used only for filings BSE files
     under "General" or leaves blank. This layer matters: 688 rows sit in
     "General", and they include RVNL's lowest-bidder and letter-of-acceptance
     notices and Reliance's Jio IPO filing, all economically real and otherwise
     scored as unclassified noise.

Anything still unmatched is labelled "Other / Unclassified" and reported with
its own count rather than folded into a convenient group.

Two subjects are marked as companions. A press release, an investor
presentation, an earnings-call transcript or a board-meeting outcome published
next to a substantive filing is the same underlying event seen from another
angle, so clustering absorbs them instead of counting them again. Priority
decides which member of a cluster names the event.
"""
from __future__ import annotations

import re

import pandas as pd

# subject -> (priority, is_companion). Lower priority wins when a cluster holds
# more than one substantive filing.
SUBJECTS = {
    "Financial Results": (1, False),
    "M&A / Strategic": (2, False),
    "Order Win": (3, False),
    "Capital & Corporate Action": (4, False),
    "Credit Rating": (5, False),
    "Management & Board Change": (6, False),
    "Board Meeting Intimation": (7, False),
    "Clarification": (8, False),
    "Board Meeting Outcome": (9, True),
    "Investor Communication": (10, True),
    "Routine Filing": (11, False),
    "Other / Unclassified": (12, False),
}

# Layer 1: exact SUBCATNAME -> subject.
SUBCAT_MAP = {
    "Financial Results": "Financial Results",
    "Integrated Filing (Financial)": "Financial Results",

    "Acquisition": "M&A / Strategic",
    "Memorandum of Understanding /Agreements": "M&A / Strategic",
    "Diversification / Disinvestment": "M&A / Strategic",
    "Joint Venture": "M&A / Strategic",

    "Award of Order / Receipt of Order": "Order Win",

    "Allotment of ESOP / ESPS": "Capital & Corporate Action",
    "Allotment of Equity Shares": "Capital & Corporate Action",
    "Record Date": "Capital & Corporate Action",
    "Dividend": "Capital & Corporate Action",
    "Dividend Updates": "Capital & Corporate Action",
    "Book Closure": "Capital & Corporate Action",
    "Sub-division / Stock Split": "Capital & Corporate Action",
    "Bonus": "Capital & Corporate Action",

    "Credit Rating": "Credit Rating",

    "Change in Management": "Management & Board Change",
    "Change in Directorate": "Management & Board Change",
    "Cessation": "Management & Board Change",
    "Retirement": "Management & Board Change",
    "Resignation of Director": "Management & Board Change",
    "Resignation of Chairman": "Management & Board Change",
    "Resignation of Company Secretary / Compliance Officer": "Management & Board Change",
    "Appointment of Company Secretary / Compliance Officer": "Management & Board Change",
    "Appointment of Statutory Auditor/s": "Management & Board Change",

    "Board Meeting": "Board Meeting Intimation",
    "Board Meeting Rescheduled": "Board Meeting Intimation",

    "Outcome of Board Meeting": "Board Meeting Outcome",
    "Outcome without intimation": "Board Meeting Outcome",

    "Analyst / Investor Meet": "Investor Communication",
    "Press Release / Media Release": "Investor Communication",
    "Press Release / Media Release (Revised)": "Investor Communication",
    "Earnings Call Transcript": "Investor Communication",
    "Investor Presentation": "Investor Communication",

    "Clarification": "Clarification",

    "Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate": "Routine Filing",
    "Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018": "Routine Filing",
    "Reg. 40 (10) - PCS Certificate for Transfer / Transmission / Transposition": "Routine Filing",
    "Reg.24(A)-Annual Secretarial Compliance": "Routine Filing",
    "Reg. 34 (1) Annual Report": "Routine Filing",
    "Business Responsibility and Sustainability Reporting (BRSR)": "Routine Filing",
    "Reg. 54 - Asset Cover details": "Routine Filing",
    "Reg. 32 (1), (3) - Statement of Deviation & Variation": "Routine Filing",
    "Newspaper Publication": "Routine Filing",
    "Closure of Trading Window": "Routine Filing",
    "AGM": "Routine Filing",
    "Postal Ballot": "Routine Filing",
    "Amendments to Memorandum & Articles of Association": "Routine Filing",
    "Monitoring Agency Report": "Routine Filing",
    "Trading Plan under SEBI (PIT) Regulations, 2015": "Routine Filing",
    "Code of Conduct under SEBI (PIT) Regulations, 2015": "Routine Filing",
    "Change in Registered Office Address": "Routine Filing",
    "Disclosures under Reg. 29(2) of SEBI (SAST) Regulations, 2011": "Routine Filing",
    "Disclosures under Reg. 29(1) of SEBI (SAST) Regulations, 2011": "Routine Filing",
}

# Layer 2: ordered keyword rules over NEWSSUB + HEADLINE, most specific first.
#
# Order is deliberate. "Financial Results" sits near the top because a results
# filing almost always also declares a dividend, and the dividend keywords would
# otherwise capture it. The narrow newspaper rule sits above it for the mirror
# reason: a newspaper advertisement *of* the results is a compliance filing, not
# the results announcement.
KEYWORD_RULES = [
    ("Routine Filing",
     r"newspaper (?:publication|clipping|advertisement)|publication of (?:the )?extract"),
    ("Financial Results",
     r"financial results|unaudited results|audited results|quarterly results|"
     r"results for the (?:quarter|period|year)"),
    ("Order Win",
     r"lowest bidder|\bl1\b|letter of acceptance|letter of award|\bloa\b|work order|"
     r"receipt of order|award of order|bags? (?:an? )?order|order worth|"
     r"contract (?:win|award|received)|emerges as"),
    ("M&A / Strategic",
     r"acquisi|acquire|amalgamat|\bmerger\b|joint venture|\bjv\b|divest|disinvest|"
     r"memorandum of understanding|\bmou\b|initial public offer|\bipo\b|demerger|"
     r"scheme of arrangement|sale of shares|sale of stake|stake sale"),
    ("Credit Rating",
     r"credit rating|rating action|\bicra\b|\bcrisil\b|care ratings"),
    ("Capital & Corporate Action",
     r"dividend|\bbonus\b|stock split|sub-?division|buy-?back|allotment|debenture|"
     r"\bncd\b|fund rais|\bqip\b|rights issue|record date|book closure|preferential issue|"
     r"stock option|\besos\b|\besop\b|stock incentive|grant of units"),
    ("Investor Communication",
     r"investor|analyst|media release|press release|earnings call|conference call|"
     r"transcript|presentation|news clipping|newspaper clipping"),
    ("Clarification",
     r"clarification|confirmation on news"),
    ("Management & Board Change",
     r"appointment|resignation|cessation|retirement|managing director|chief executive|"
     r"\bcfo\b|\bceo\b|company secretary|key managerial|\bkmp\b"),
    ("Routine Filing",
     r"loss of certificate|duplicate certificate|reg\.? ?74|regulation 74|trading window|"
     r"compliance certificate|annual report|\bbrsr\b|postal ballot|annual general meeting|"
     r"\bagm\b|code of conduct|re-?lodgement|dematerialisation|secretarial|"
     r"related party transaction|regulation 23\(9\)|regulation 6\(1\)|sebi circular|"
     r"non-applicability|scrutinizer|voting result"),
]

_COMPILED = [(subj, re.compile(pat, re.IGNORECASE)) for subj, pat in KEYWORD_RULES]


#: "Outcome of Board Meeting" names the meeting, not the news. HDFC Bank files
#: its quarterly results under this label with no separate Financial Results
#: row, so taking the label at face value would drop two earnings events from
#: the PEAD sample. Filings carrying it are re-read for their actual content and
#: promoted when a substantive subject is found.
CONTAINER_SUBJECTS = {"Board Meeting Outcome"}
_SUBSTANTIVE_MAX_PRIORITY = 8


def classify(ann: pd.DataFrame) -> pd.DataFrame:
    """Attach subject, priority, companion flag and the rule that fired."""
    out = ann.copy()
    text = (
        out["NEWSSUB"].fillna("") + " || " + out["HEADLINE"].fillna("")
    ).str.replace(r"\s+", " ", regex=True)

    subject = out["SUBCATNAME"].map(SUBCAT_MAP)
    rule = pd.Series("subcategory", index=out.index).where(subject.notna())

    # Promote container labels whose text names a substantive subject.
    container = subject.isin(CONTAINER_SUBJECTS)
    for subj, rx in _COMPILED:
        if SUBJECTS[subj][0] > _SUBSTANTIVE_MAX_PRIORITY:
            continue
        hit = container & text.str.contains(rx, na=False)
        subject = subject.mask(hit, subj)
        rule = rule.mask(hit, "outcome-resolved:" + subj)
        container = subject.isin(CONTAINER_SUBJECTS)

    unresolved = subject.isna()
    for subj, rx in _COMPILED:
        hit = unresolved & text.str.contains(rx, na=False)
        subject = subject.mask(hit, subj)
        rule = rule.mask(hit, "keyword:" + subj)
        unresolved = subject.isna()

    out["subject"] = subject.fillna("Other / Unclassified")
    out["subject_rule"] = rule.fillna("unmatched")
    out["subject_priority"] = out["subject"].map(lambda s: SUBJECTS[s][0])
    out["is_companion"] = out["subject"].map(lambda s: SUBJECTS[s][1])
    out["fiscal_period"] = extract_fiscal_period(text).where(
        out["subject"].eq("Financial Results")
    )
    return out


# Results filings name the period they report on: "for the Quarter Ended June
# 30, 2024". QUARTER_ID is null in all 2,530 rows, so the text is the only
# handle on it.
_PERIOD_PATTERNS = [
    re.compile(r"end(?:ed|ing)\b[^A-Za-z0-9]{0,12}([A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
    re.compile(r"end(?:ed|ing)\b[^A-Za-z0-9]{0,12}(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]{2,8},?\s+\d{4})", re.IGNORECASE),
]


def extract_fiscal_period(text: pd.Series) -> pd.Series:
    """Return the reporting period end date named in each filing, if any.

    Used to keep re-filings of one quarter -- notably the "in machine-readable
    form" copies HDFC Bank publishes several days later -- attached to the
    original event instead of counting as a fresh earnings release.
    """
    def parse(s: str):
        for rx in _PERIOD_PATTERNS:
            m = rx.search(s or "")
            if m:
                dt = pd.to_datetime(m.group(1).replace(",", ""), errors="coerce",
                                    dayfirst=not m.group(1)[0].isalpha())
                if pd.notna(dt):
                    return dt.strftime("%Y-%m-%d")
        return None

    return text.map(parse)
