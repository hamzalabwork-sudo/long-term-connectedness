# How long is long-term connectedness? Centring, lag structure and window length in TVP-VAR frequency spillovers

Replication code for:

> Zhang, Z., Dangado, S. S., Ibrahim, H. *How long is long-term connectedness? Centring, lag structure and window length in TVP-VAR frequency spillovers.* Manuscript submitted to *International Review of Economics and Finance*.

Frequency connectedness from TVP-VARs (Diebold–Yilmaz; Baruník–Křehlík; Antonakakis et al.) is widely used to claim that long-term spillovers dominate short-term ones. The paper asks what the long-term band measures. It shows that the band mixes genuine slow spillovers with two artefacts:
- full-sample centring, the default of the R package [ConnectednessApproach](https://cran.r-project.org/package=ConnectednessApproach);
- short lag structures and forgetting windows that cannot represent slow dynamics.

The paper then:
- derives what the estimators converge to and a bias decomposition;
- chooses the decay rates by predictive likelihood;
- proposes a TVP-HAR-VAR;
- estimates by indirect inference how much of the slow movement in hourly crypto volatility is level drift.

## Layout
```
code/      data download and cleaning, TVP-VAR connectedness (a validated Python port of
           ConnectednessApproach), replication of Shi & Lin (2026), centring analysis,
           package and implied-volatility datasets, simulations, pre-specified forecasts
iref/      the paper's new analyses
  code/    population connectedness, pseudo-true values, decay-rate selection, standard Kalman
           filter, TVP-HAR-VAR, hourly vs daily, window trade-off, indirect inference, exhibits
  results/ result files behind every table and figure of the paper
work/      working directory for code/ (data are downloaded here; see work/data/README.md)
  results/ result summaries of code/
notes/     pre-specified forecasting design and its SHA-256 hash log
```

## Requirements
```
pip install -r requirements.txt     # numpy, pandas, scipy, torch, matplotlib, statsmodels
```
- **R** (with `zoo`, `progress`, `MASS`) is needed only to validate the Python port against the package and to convert the package's example datasets.
- **A GPU** is optional. The scripts take `--device` / `--devices` (`cpu` by default, or e.g. `cuda:0`).

## Reproducing the results

### 1. Data, replication and centring (run from `work/`)
```bash
cd work
for ex in binance okx bitget bitfinex bybit; do python ../code/full_download.py --exchange $ex; done
python ../code/build_rv.py                                   # hourly log RV (connectedness panel)
python ../code/build_rv.py --max-gap 0 --tag causal          # strictly causal panel (forecasts, decay selection)
python ../code/build_rv.py --sample-minutes 15 --tag rv15    # 15-minute sampling

python ../code/run_connectedness.py --mode package --name shilin_package
python ../code/run_connectedness.py --mode realtime --prior-hours 720 --demean ewma --demean-halflife 360 --name shilin_ewma15d
python ../code/compare_shilin.py --runs shilin_package shilin_ewma15d
bash ../code/robustness_grid.sh                              # incl. matched EWMA half-lives
python ../code/package_datasets.py --data pkgdata/csv --out results/pkgdata
python ../code/build_ivol.py && python ../code/package_datasets.py --data ivoldata/csv --out results/ivol
python ../code/sim_centering.py --panel data/processed/logrv_ds.csv.gz --out results/sim/centering_level30d.csv
python ../code/sim_centering.py --panel data/processed/logrv_ds.csv.gz --dgp harvar --scales 0 --seeds 5 --out results/sim/centering_harvar.csv
```
The pre-specified forecasting exercise (Section 5.6 of the paper) is run with `code/forecast.py`, `code/evaluate.py` and `code/test_stage.py`; see their headers and `notes/`.

### 2. The paper's new analyses (run from `iref/`)
```bash
cd iref
python code/check_spectral.py                 # population connectedness vs finite-H implementation
python code/pseudo_true.py                    # pseudo-true values vs Monte Carlo (Table 4, Panel A)
python code/check_local_sampling.py           # local-sampling term
python code/predict_gap.py                    # predicted vs estimated centring gap (Table 4, Panel B)

python code/select_halflife.py                # decay rates by predictive log score (Table 5)
python code/select_halflife.py --freqs daily --h-coef 15 30 60 120 240 365 730 --h-vol 7 15 30 60 120 240 365 --out results/select_halflife_daily_ext.csv
python code/select_halflife.py --freqs hourly --h-coef 7 15 30 --h-vol 0.25 0.5 1 2 3 5 --out results/select_halflife_hourly_ext.csv
python code/score_tests.py                    # Diebold-Mariano-type tests of score differences
python code/select_q.py                       # standard Kalman filter with estimated state covariance

python code/hourly_vs_daily.py                # Fig. 2
python code/aggregation.py
python code/sim_aggregation.py

python code/run_harvar.py                     # TVP-HAR-VAR vs TVP-VAR(3)
python code/sim_harvar.py --worlds harvar drift --hl-coef 15 60 120 --unstable carry --out results/sim_harvar_hl.csv   # Fig. 3
python code/real_estimates.py                 # Table 6

for n in exchange:BN instrument:BTCS instrument:BTCP instrument:ETHS instrument:ETHP exchange:OKX exchange:BG exchange:BF; do
  d=results/sim_indirect_${n/:/_}
  python code/sim_indirect.py --network $n --seeds 2 --out $d     # Binance in the paper: --seeds 3
  python code/match_indirect.py --network $n --sim $d/estimates.csv --truth $d/truth.csv
done                                          # Table 7, Fig. 5

python code/pairwise.py                       # pairwise connectedness matrices
python code/network_figs.py                   # Fig. 4, Table S11, Fig. S1
python code/make_data_exhibits.py             # Table 1, Fig. 1
python code/make_exhibits.py                  # Tables 4-7, S6 and Figs. 2, 3, 5 (written to paper/)
```
Most steps take minutes on a workstation. The simulations (`sim_harvar.py`, `sim_indirect.py`) take one to two hours per run with 10-15 worker processes.

## Validation against the R package
`code/connectedness.py` ports the package's `TVPVAR`, `BayesPrior`, `FEVD` and `FrequencyConnectedness` step by step. It reproduces the R code to within 3×10⁻¹⁰ in the coefficients and 6×10⁻⁶ in all connectedness measures. See `code/validate_r.py` and `code/fetch_rsrc.sh`.

## Data
- **Crypto data:** collected from the public market-data interfaces of Binance, OKX, Bitget, Bitfinex and Bybit with `code/full_download.py`. Raw data are not redistributed.
- **Implied-volatility indices:** from FRED.
- **Other datasets:** distributed with ConnectednessApproach.

See `work/data/README.md`.

## License
MIT (see `LICENSE`).
