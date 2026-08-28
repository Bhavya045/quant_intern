# Data placement

The supplied datasets are **not** committed to this repository (see `.gitignore`).

To reproduce the analysis, place the five 1-minute OHLCV files and the
announcement files in `data/raw/`:

```
data/raw/
├── corporate_announcements.csv
├── corporate_announcements.jsonl
├── metadata.json
├── RELIANCE.csv
├── HDFCBANK.csv
├── NYKAA.csv
├── HAL.csv
└── RVNL.csv
```

Alternatively, point the pipeline at the files wherever they already live:

```bash
export QAI_DATA_DIR="/path/to/Quant Analyst Intern"   # bash
$env:QAI_DATA_DIR = "C:\path\to\Quant Analyst Intern" # PowerShell
```

`src/config.py` resolves the data directory in this order:

1. the `QAI_DATA_DIR` environment variable, if set;
2. `data/raw/` inside this repository;
3. `../Quant Analyst Intern/` next to this repository.

## Expected files

| File | Contents |
|---|---|
| `corporate_announcements.csv` / `.jsonl` | 2,530 BSE announcements for 5 companies, 2023-08-20 to 2026-08-20 |
| `<SYMBOL>.csv` | 1-minute OHLCV bars, ~277,510 rows each, 744 trading sessions, 2023-08-21 to 2026-08-19 |
| `metadata.json` | Scrip-code mapping, field list, row counts, SHA-256 checksums |

## Note on `DATA_FORMAT.md`

The assessment brief references a `DATA_FORMAT.md` describing field definitions,
the timestamp convention and data caveats. That file was **not** included in the
supplied bundle. All timestamp and field assumptions used here are therefore
derived from the data itself and from `metadata.json`, and are documented
explicitly in the README and report.
