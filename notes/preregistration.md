# Pre-registered forecasting analysis

Frozen on 2026-10-21, before any forecast was evaluated on the test period. The SHA-256 of this file is logged in `notes/preregistration.sha256` and on s28 (`~/frl_work/logs/prereg.log`). Any later change is appended in the change log at the end, with its date and reason. The test period is evaluated once, after the specification is frozen.

**Already seen at freeze time.** Descriptive connectedness for 2021–2025 (replication of Shi & Lin, 2026) and the centring simulation. No forecast has been fitted or evaluated on any period.

## 1. Data
- **Series:** hourly deseasonalised log realised variance, `logrv_ds` (code: `build_rv.py` with defaults).
  - 16 series: {BTC, ETH} × {spot, perpetual} × {Binance, OKX, Bitget, Bitfinex}.
  - Built from 5-minute returns.
  - Hour-of-day seasonality is estimated on the training period only.
- **RV in levels:** `RV_ds = exp(logrv_ds)`.

## 2. Sample splits (UTC)
| Period | Dates | Use |
|---|---|---|
| Burn-in | 2021-01-01 to 2021-01-30 (720 h) | Prior and initial centring of the real-time TVP-VAR |
| Training | 2021-01-31 to 2023-12-31 | Estimation |
| Validation | 2024-01-01 to 2024-12-31 | Specification choices only |
| Test | 2025-01-01 to 2026-09-30 | Sealed; evaluated once |

## 3. Targets
For horizons h ∈ {1, 8, 24, 168} hours:

z_t(h) = log( (1/h) · Σ_{j=1..h} RV_ds,t+j )

## 4. Predictors
All predictors are known at the end of hour t.
- **HAR (own series):** log RV_ds at t, and its means over the last 24, 168 and 720 hours.
- **Cross-market (information-matched):** for each of the 6 network partners of the series, log RV at t and the mean over the last 24 hours (12 regressors). The partners are the 3 other venues of the same instrument and the 3 other instruments on the same venue.
- **Connectedness, primary version:** real-time TVP-VAR (`rt_ewma15d_true`).
  - Settings: causal EWMA centring with a 15-day half-life; prior from the burn-in window; 3 lags; half-lives 15 and 7 days; H = 720.
  - Bands are defined by **true** cycle length: short 2–8 h, mid 8–24 h, long > 24 h.
  - For series i, from both of its networks: FROM_i, TO_i and the network TCI. That gives 6 regressors per domain, in the time domain and in each band.
- **Connectedness comparators:**
  - `rt_tvint_true`: time-varying intercept.
  - `pkg_full_true`: package centring and prior over the full sample through 2026-09. This is infeasible in real time, so it is reported as a look-ahead benchmark only.

## 5. Models
Estimation is OLS on a direct h-step target with an expanding window. Models are re-estimated at the start of each calendar month on all observations whose target is fully observed at the forecast origin.

| Model | Regressors |
|---|---|
| M0 | HAR |
| **M1** | HAR + cross-market (**benchmark**) |
| M2 | M1 + time-domain connectedness |
| M3_b | M1 + band-b connectedness, b ∈ {short, mid, long} |
| M4 | M1 + all three bands |

## 6. Losses
- **Primary: QLIKE on levels.** L = R/F − log(R/F) − 1.
  - R = (1/h) Σ_j RV_ds,t+j.
  - F = exp(ẑ + σ̂²/2), where σ̂² is the in-sample residual variance.
- **Secondary:** MSE of z.

## 7. Hypotheses
- **H1 (incremental value):** for some band b and horizon h, M3_b has lower pooled loss than M1.
- **H2 (horizon matching):** the band with the largest gain over M1 shifts from short towards long as h grows.
  - Short is best at h = 1; long is best at h ∈ {24, 168}.
  - Placebo: at each h, the matched band beats the mismatched bands.
- **H3 (centring):** band measures from real-time centring forecast at least as well as those from package centring computed in real time. The `pkg_full` look-ahead benchmark is reported separately.

## 8. Inference
- **Pooled test:** for each (model, h), take the cross-sectional mean loss differential against M1, d̄_t = (1/16) Σ_i (L_M1,i,t − L_M,i,t).
  - Diebold–Mariano t-statistic with Newey–West, lags = max(h, ⌊4 (T/100)^{2/9}⌋).
  - Test is one-sided: H_A is that M improves on M1.
- **Per series:**
  - Clark–West (2007) for nested models on z.
  - Diebold–Mariano on QLIKE.
- **Multiple testing:** Holm correction at 5% across the 12 pooled tests (3 bands × 4 horizons) for H1, and the same for H2's matched-vs-mismatched comparisons.
- **Model Confidence Set** (Hansen, Lunde & Nason, 2011) at 90% over {M0, M1, M2, M3_short, M3_mid, M3_long, M4} for each h, on pooled QLIKE, using a block bootstrap with block length 24 and 5,000 replications. Test stage only.

## 9. Economic value (test stage)
- **Strategy:** volatility targeting on the Binance BTC and ETH perpetuals, rebalanced every 24 h.
  - Weight w_t = σ* / σ̂_t(24), capped at 3.
  - σ* is the training-period average volatility.
  - Cost: 5 bp per unit of turnover.
- **Report:**
  - realised-volatility tracking error versus σ*;
  - Sharpe ratio;
  - Fleming, Kirby & Ostdiek (2001) performance fee for models versus M1, with relative risk aversion γ ∈ {1, 3}.

## 10. Event windows (test stage, descriptive)
Pooled loss differentials over ±7 days around:
- the Bybit hack (2025-02-21);
- the liquidation cascade (2025-10-10).

## 11. Decision rules
- Only validation-period results may change the specification: lag sets, which connectedness measures enter, or regularisation if OLS is unstable.
- Every change is logged below with a date.
- All pre-registered results are reported whatever their sign. A null result on H1/H2 is reported as such.

## Change log
- 2026-10-21: frozen.
- 2026-10-21, after the validation run, **exploratory addition only**.
  - New models M0+<domain>: univariate HAR plus connectedness, without cross-market lags. Each is compared with M0.
  - Purpose: to separate the cross-market information that connectedness measures inherit from any incremental content. Most of the literature compares with univariate HAR.
  - Pre-registered models, hypotheses and tests are unchanged. The new comparison is reported as exploratory.
  - Validation outcome that motivated it: no M3_b beats M1 at any h (Holm p = 1.00), and there is no matched-band advantage. Cross-market lags beat HAR at h = 1 (+0.76%, t = 5.3).
- 2026-10-21, before unsealing the test, **new secondary hypothesis H4**.
  - Motivation: connectedness levels (TCI, FROM, TO) are a weak use of the network. The economically meaningful channel is which partner spills over to series i.
  - Models:
    - E0 = HAR + equal-weighted partner log RV (lag 1 h and 24-h mean, in the instrument and the exchange network; 4 regressors).
    - S_d = HAR + the same four regressors, with partners weighted by their FEVD shares in row i of domain d (time or one band).
  - H4: S_d has lower pooled QLIKE than E0. Same Diebold–Mariano test as section 8, Holm correction over the 4 domains × 4 horizons.
  - Validation results for H4 are recorded below before the test is run.
- 2026-10-21, **H4 validation (2024) results**, recorded before unsealing.
  - S_d beats E0 in all 16 cells (4 domains × 4 horizons). Gains are +0.02–0.03% at h = 1, +0.05–0.06% at h = 8, +0.10–0.11% at h = 24 and +0.30–0.36% at h = 168.
  - DM t ranges from 1.0 to 2.2; Holm-adjusted p ranges from 0.25 to 0.78.
  - Band weights perform the same as time-domain weights.
  - E0 versus M0: +0.80% at h = 1 (t = 5.4).
  - No further specification changes; the specification is final for the test.
- 2026-10-21, after unsealing (test evaluated once at 10:19:57Z), **deviation: correction of a look-ahead**, recorded before rerunning.
  - Problem: an internal review found that the "real-time" pipeline linearly interpolated missing hours in three places: the panel (gaps of 3 h or less), the TVP-VAR input, and the forecast features. That uses one future value for 326 of 806,016 cells (0.04%).
  - Correction:
    - The panel is rebuilt without interpolation (`build_rv.py --max-gap 0`, tag `causal`).
    - Real-time TVP-VAR inputs and forecast features now carry the last value forward.
    - The connectedness runs are recomputed as `rtc_*` / `pkgc_*`, and validation and test forecasts are rerun once under the unchanged specification.
  - Reporting: the corrected results are reported, and the original results are kept and disclosed. No other change.
- 2026-10-21, **reporting additions requested by the internal review**. These are computed from the same losses and do not change the specification.
  - Pooled MSE gains and Clark–West statistics, reported alongside QLIKE.
  - 95% confidence intervals for matched-versus-mismatched band differences.
  - Holm correction for H4, as specified; the earlier table used Bonferroni.
  - Block-bootstrap intervals for performance fees.
  - The γ = 1 fees, tracking error and the Bybit window, which were computed but not shown.
