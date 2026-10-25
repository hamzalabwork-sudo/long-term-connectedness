# Frequency connectedness in crypto spot and perpetual markets: code

Every step runs from a work directory that contains `data/` and `results/`. Paths below are relative to that directory, and the scripts live in `../code`.

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Download 1-minute klines (free public APIs: Binance, OKX, Bitget, Bitfinex, Bybit), Jan 2021 – Sep 2026 | `full_download.py --exchange <ex>` (resumable; `--reverse` runs a second process from the end) | `data/raw/<ex>/*.csv.gz` |
| 2. Completeness check | `check_download.py data/raw` | console |
| 3. Hourly log realised variance | `build_rv.py` | `data/processed/logrv_ds.csv.gz`, `stale_share.csv.gz`, `diagnostics.json` |
| 4. TVP-VAR time/frequency connectedness | `run_connectedness.py --name <run> [options]` | `results/conn/<run>/` |
| 5. Replication vs Shi & Lin (2026) Tables 2–3 | `compare_shilin.py --runs <run> ...` | console |
| 6. Centring mechanism | `exp_centering.py`, `centering_diagnostic.py`, `sim_centering.py` | console, `results/sim/*.csv` |
| 7. Forecasts (pre-registered) | `forecast.py --split validation --conn <runs>` | `results/forecast/<split>/losses.npz` |
| 8. Inference | `evaluate.py --split validation` | `results/forecast/<split>/tests*.csv` |
| 9. Bitfinex staleness | `build_rv.py --sample-minutes 15 --tag rv15`, then `staleness.py` | `results/staleness/*.csv` |
| 10. Robustness grid (19 runs × 8 networks) | `robustness_grid.sh` (and `sync_grid.sh` to share runs with a second machine) | `results/conn/rob_*` |
| 11. Mechanism checks (network-average inference, pseudo-true VAR(1)) | `mechanism_checks.py` | console |
| 12. Test-stage analyses (MCS, per-series tests, economic value with bootstrap CIs, events) | `test_stage.py --split test` | `results/forecast*/test/stage_*.csv` |
| 13. Paper tables and figures | `make_tables.py`, `make_figures.py` | `../paper/tables`, `../paper/figs` |

### Generality and extra analyses
| Step | Script | Output |
|---|---|---|
| Centring comparison on the 17 example datasets of ConnectednessApproach (.rda converted to CSV with R) | `package_datasets.py --data pkgdata/csv` | `results/pkgdata/summary.csv` |
| Equivalence bounds for forecast gains; network-role flips between centrings | `bounds_and_flips.py` | `results/extra/` |

### Building the paper
Compile `supplementary.tex` first, then run `python ../code/make_suppnum.py supplementary.aux suppnum.tex`, then compile `main.tex`. The manuscript cites the supplementary tables through the generated `suppnum.tex`, so it needs neither the `xr` package nor a particular build order on Overleaf. Rerun `make_suppnum.py` whenever the order of supplementary tables changes.

### Corrected real-time pipeline (used in the paper)
```
python ../code/build_rv.py --max-gap 0 --tag causal          # no interpolation
python ../code/run_connectedness.py --panel data/processed/logrv_ds_causal.csv.gz \
    --end "2026-09-30 23:00" --mode realtime --prior-hours 720 --demean ewma --demean-halflife 360 \
    --true-periods --name rtc_ewma15d_true                    # plus rtc_window_true, rtc_tvint_true, pkgc_full_true
FRL_UNSEAL_TEST=1 python ../code/forecast.py --split test --panel data/processed/logrv_ds_causal.csv.gz \
    --conn rtc_ewma15d_true rtc_window_true rtc_tvint_true pkgc_full_true --out results/forecast_causal
python ../code/evaluate.py --root results/forecast_causal --split test
python ../code/test_stage.py --root results/forecast_causal --split test
```
In real-time mode, missing values are carried forward, never interpolated. The original pre-correction results are kept in `results/forecast/`.

### Simulation designs (`sim_centering.py`)
- `--dgp drift`: a VAR fitted to the de-levelled data, plus a level. The level is either the observed slow level or, with `--level-mode random`, a random common AR(1) level.
- `--dgp persistent`: a VAR(3) on the raw series.
- `--dgp harvar`: a HAR-type VAR(168) on the raw series, giving slow genuine spillovers.

## Cleaning rules (`build_rv.py`)
- **Zero-volume minutes carry no trade.** Exchanges repeat the last price in them, including during outages, e.g. OKX perpetuals on 2021-01-10 19:27–21:14.
- **Bad prints:** a deviation of more than 3% from the cross-venue median of the same asset, lasting 3 minutes or less. Longer excursions are genuine dislocations and are kept, e.g. Binance ETH perpetual on 2021-05-19 (34 minutes, up to 21%).
- **Previous-tick fill** for at most 15 minutes; returns spanning longer gaps are dropped.
- **Hourly RV** from 5-minute returns, rescaled when a few returns are missing. Hour-of-day seasonality is estimated on the training period (to 2023-12-31) only.

## Connectedness (`connectedness.py`)
- **Validation:** a step-by-step port of ConnectednessApproach (Gabauer; CRAN) `TVPVAR`, `BayesPrior`, `FEVD` (generalised, time and frequency) and `FrequencyConnectedness` (ABS).
  - It was checked against the package's R code, sourced from CRAN, by `validate_r.py` / `validate_r.R`.
  - Coefficients match to ≤ 3e-10 and all connectedness measures to ≤ 5e-6, including the stability fallback.
- **Additions:**
  - `mode="realtime"`: prior and centring from an initial window only.
  - Causal EWMA centring (`demean="ewma"`).
  - A time-varying intercept (`intercept=True`).
  - Band edges in true cycle length (`--true-periods`). The package labels bands by π/ω, so its "1–8" band covers true cycles of 2–16 observations.
- **Memory:** only the current state covariance is kept. The package stores all of them, which is ~236 GB at k=16, 3 lags, 44k hours.

## Reproduced numbers (package settings, 2021–2025)
- **Total connectedness** is within 0.5 points of Shi & Lin in all 8 networks, and the three bands are within about 1 point.
- **NET signs** match in 31 of 32 cases.

## Pre-registration
- `../notes/preregistration.md`, frozen 2026-10-21, SHA-256 in `../notes/preregistration.sha256`.
- `forecast.py --split test` refuses to run unless `FRL_UNSEAL_TEST=1`.
