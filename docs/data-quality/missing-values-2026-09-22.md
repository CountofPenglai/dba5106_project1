# Missing-value audit — 2026-09-22

Source: `100_Portfolios_10x10_Daily.csv`. Counts use raw values before column exclusion, missing-value replacement, or division by 100.

The file identifies **-99.99 and -999** as missing-value markers. Exact **-99 is counted separately** and is not automatically classified as missing. There are no exact -99 or -999 observations in this file. The active model reads only the equal-weighted daily returns section.

## Full-file sections

| Section | Rows | Columns | -99 | -99.99 | -999 |
|---|---:|---:|---:|---:|---:|
| Average Value Weighted Returns -- Daily | 26,296 | 100 | 0 | 35,246 | 0 |
| Average Equal Weighted Returns -- Daily | 26,296 | 100 | 0 | 35,246 | 0 |
| Number of Firms in Portfolios | 26,296 | 100 | 0 | 0 | 0 |
| Average Firm Size | 26,296 | 100 | 0 | 35,291 | 0 |

Coverage: **1926-07-01 to 2026-07-31**, 26,296 observations per section. Column numbers below exclude the date column.

## All 100 columns in the model's source section

All columns have zero exact -99, zero -999, and zero blank values. This table therefore displays the -99.99 count. The machine-readable CSV retains separate columns for all three markers and blanks.

| Column | Asset | -99.99 count | Missing % | Last marker date | Default treatment |
|---|---|---:|---:|---|---|
| 1 | SMALL LoBM | 2,548 | 9.6897% | 1962-06-29 | Keep; markers become NaN |
| 2 | ME1 BM2 | 3,181 | 12.0969% | 1961-02-09 | Keep; markers become NaN |
| 3 | ME1 BM3 | 2,902 | 11.0359% | 1963-06-28 | Keep; markers become NaN |
| 4 | ME1 BM4 | 1,494 | 5.6815% | 1944-05-06 | Keep; markers become NaN |
| 5 | ME1 BM5 | 939 | 3.5709% | 1953-01-28 | Keep; markers become NaN |
| 6 | ME1 BM6 | 305 | 1.1599% | 1932-03-22 | Keep; markers become NaN |
| 7 | ME1 BM7 | 297 | 1.1294% | 1929-06-29 | Keep; markers become NaN |
| 8 | ME1 BM8 | 3 | 0.0114% | 1930-01-27 | Keep; markers become NaN |
| 9 | ME1 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 10 | SMALL HiBM | 0 | 0.0000% | — | Keep; markers become NaN |
| 11 | ME2 BM1 | 2,580 | 9.8114% | 1958-03-17 | Keep; markers become NaN |
| 12 | ME2 BM2 | 1,485 | 5.6472% | 1959-10-20 | Keep; markers become NaN |
| 13 | ME2 BM3 | 689 | 2.6202% | 1944-02-05 | Keep; markers become NaN |
| 14 | ME2 BM4 | 8 | 0.0304% | 1938-10-27 | Keep; markers become NaN |
| 15 | ME2 BM5 | 303 | 1.1523% | 1934-02-20 | Keep; markers become NaN |
| 16 | ME2 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 17 | ME2 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 18 | ME2 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 19 | ME2 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 20 | ME2 BM10 | 0 | 0.0000% | — | Keep; markers become NaN |
| 21 | ME3 BM1 | 614 | 2.3350% | 1948-08-30 | Keep; markers become NaN |
| 22 | ME3 BM2 | 604 | 2.2969% | 1936-06-30 | Keep; markers become NaN |
| 23 | ME3 BM3 | 293 | 1.1142% | 1935-12-18 | Keep; markers become NaN |
| 24 | ME3 BM4 | 292 | 1.1104% | 1930-06-30 | Keep; markers become NaN |
| 25 | ME3 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 26 | ME3 BM6 | 299 | 1.1371% | 1927-06-30 | Keep; markers become NaN |
| 27 | ME3 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 28 | ME3 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 29 | ME3 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 30 | ME3 BM10 | 0 | 0.0000% | — | Keep; markers become NaN |
| 31 | ME4 BM1 | 317 | 1.2055% | 1958-06-26 | Keep; markers become NaN |
| 32 | ME4 BM2 | 5 | 0.0190% | 1933-08-04 | Keep; markers become NaN |
| 33 | ME4 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 34 | ME4 BM4 | 299 | 1.1371% | 1927-06-30 | Keep; markers become NaN |
| 35 | ME4 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 36 | ME4 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 37 | ME4 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 38 | ME4 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 39 | ME4 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 40 | ME4 BM10 | 0 | 0.0000% | — | Keep; markers become NaN |
| 41 | ME5 BM1 | 301 | 1.1447% | 1935-10-16 | Keep; markers become NaN |
| 42 | ME5 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 43 | ME5 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 44 | ME5 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 45 | ME5 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 46 | ME5 BM6 | 2 | 0.0076% | 1930-05-23 | Keep; markers become NaN |
| 47 | ME5 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 48 | ME5 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 49 | ME5 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 50 | ME5 BM10 | 1 | 0.0038% | 1927-04-30 | Keep; markers become NaN |
| 51 | ME6 BM1 | 0 | 0.0000% | — | Keep; markers become NaN |
| 52 | ME6 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 53 | ME6 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 54 | ME6 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 55 | ME6 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 56 | ME6 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 57 | ME6 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 58 | ME6 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 59 | ME6 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 60 | ME6 BM10 | 395 | 1.5021% | 1936-05-07 | Keep; markers become NaN |
| 61 | ME7 BM1 | 0 | 0.0000% | — | Keep; markers become NaN |
| 62 | ME7 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 63 | ME7 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 64 | ME7 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 65 | ME7 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 66 | ME7 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 67 | ME7 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 68 | ME7 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 69 | ME7 BM9 | 0 | 0.0000% | — | Keep; markers become NaN |
| 70 | ME7 BM10 | 852 | 3.2400% | 2001-06-29 | Keep; markers become NaN |
| 71 | ME8 BM1 | 0 | 0.0000% | — | Keep; markers become NaN |
| 72 | ME8 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 73 | ME8 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 74 | ME8 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 75 | ME8 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 76 | ME8 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 77 | ME8 BM7 | 253 | 0.9621% | 1955-06-30 | Keep; markers become NaN |
| 78 | ME8 BM8 | 0 | 0.0000% | — | Keep; markers become NaN |
| 79 | ME8 BM9 | 1 | 0.0038% | 1931-03-12 | Keep; markers become NaN |
| 80 | ME8 BM10 | 1,195 | 4.5444% | 1940-06-29 | Keep; markers become NaN |
| 81 | ME9 BM1 | 0 | 0.0000% | — | Keep; markers become NaN |
| 82 | ME9 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 83 | ME9 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 84 | ME9 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 85 | ME9 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 86 | ME9 BM6 | 297 | 1.1294% | 1928-06-30 | Keep; markers become NaN |
| 87 | ME9 BM7 | 2 | 0.0076% | 1933-08-04 | Keep; markers become NaN |
| 88 | ME9 BM8 | 251 | 0.9545% | 1954-06-30 | Keep; markers become NaN |
| 89 | ME9 BM9 | 253 | 0.9621% | 1960-06-30 | Keep; markers become NaN |
| 90 | ME9 BM10 | 2,327 | 8.8493% | 2025-06-30 | Drop entire column |
| 91 | BIG LoBM | 0 | 0.0000% | — | Keep; markers become NaN |
| 92 | ME10 BM2 | 0 | 0.0000% | — | Keep; markers become NaN |
| 93 | ME10 BM3 | 0 | 0.0000% | — | Keep; markers become NaN |
| 94 | ME10 BM4 | 0 | 0.0000% | — | Keep; markers become NaN |
| 95 | ME10 BM5 | 0 | 0.0000% | — | Keep; markers become NaN |
| 96 | ME10 BM6 | 0 | 0.0000% | — | Keep; markers become NaN |
| 97 | ME10 BM7 | 0 | 0.0000% | — | Keep; markers become NaN |
| 98 | ME10 BM8 | 809 | 3.0765% | 2000-06-30 | Keep; markers become NaN |
| 99 | ME10 BM9 | 2,247 | 8.5450% | 2001-06-29 | Keep; markers become NaN |
| 100 | BIG HiBM | 6,603 | 25.1103% | 2026-07-31 | Drop entire column |

## Current handling and impact

The active preprocessing module is `portfolio/data/preprocessing.py`. It drops
the two excluded columns first, replaces official missing markers and converts
percentage-point returns to decimals. It does not impute values or silently
drop rows. Training rejects missing values in the used history.

After exclusion, the full historical panel still has 26,316 missing cells
across 35 retained assets; the last is June 29, 2001. Active report commands
retain August 2, 2021–July 31, 2026: **1,255 dates × 98 assets, zero missing
cells**. A daily fit uses only its specified trailing window. The earlier
recent-period audit (June 27, 2024–July 31, 2026) also found no remaining
missing cells. Exact -99 is distinct from the official markers and does not
occur in this input file.

The recent original audit is preserved under `results/data_audit/recent/` with
unchanged source fingerprints. Full-history and since-2000 audit files are in the
external backup described in [the results guide](../results.md).

## Reproduce

```bash
.venv/bin/python audit_missing_values.py --start 2021-08-01 --end 2026-07-31
.venv/bin/python audit_missing_values.py --output results/runs/new_missing_audit
```

An output directory contains `counts.csv` (all 400 section/asset records) and
`audit.json`. The audit does not modify the raw file or preprocessing policy.

Source SHA-256: `11399a830c7b2a192d652b39e866a74c05d7380dd6edf1573ce1c40f9d271c23`.
