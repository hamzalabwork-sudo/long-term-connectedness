#!/usr/bin/env bash
# Robustness of the centring result: one setting changed at a time around the Shi & Lin
# baseline (3 lags, H = 720, half-lives 15/7 days, package band labels, 5-minute RV),
# each with package centring and with causal EWMA centring, on all 8 networks (2021-2025).
# Run from the work directory:  bash ../code/robustness_grid.sh
set -u
PKG="--mode package"
EWMA="--mode realtime --prior-hours 720 --demean ewma --demean-halflife 360"
P5="data/processed/logrv_ds.csv.gz"
P15="data/processed/logrv_ds_rv15.csv.gz"

jobs=(
  "rob_lag2_pkg|$P5|$PKG --nlag 2"            "rob_lag2_ewma|$P5|$EWMA --nlag 2"
  "rob_lag4_pkg|$P5|$PKG --nlag 4"            "rob_lag4_ewma|$P5|$EWMA --nlag 4"
  "rob_H360_pkg|$P5|$PKG --nfore 360"         "rob_H360_ewma|$P5|$EWMA --nfore 360"
  "rob_H1440_pkg|$P5|$PKG --nfore 1440"       "rob_H1440_ewma|$P5|$EWMA --nfore 1440"
  "rob_kap099_pkg|$P5|$PKG --half-lives 2.874 2.874"  "rob_kap099_ewma|$P5|$EWMA --half-lives 2.874 2.874"
  "rob_kap30_pkg|$P5|$PKG --half-lives 30 15" "rob_kap30_ewma|$P5|$EWMA --half-lives 30 15"
  # forgetting changed with the EWMA half-life matched to the coefficient half-life (reported in Table S3)
  "rob_kap099_ewmaM|$P5|--mode realtime --prior-hours 720 --demean ewma --demean-halflife 69 --half-lives 2.874 2.874"
  "rob_kap30_ewmaM|$P5|--mode realtime --prior-hours 720 --demean ewma --demean-halflife 720 --half-lives 30 15"
  "rob_true_pkg|$P5|$PKG --true-periods"      "rob_true_ewma|$P5|$EWMA --true-periods"
  "rob_rv15_pkg|$P15|$PKG"                    "rob_rv15_ewma|$P15|$EWMA"
  "rob_ewma7|$P5|--mode realtime --prior-hours 720 --demean ewma --demean-halflife 168"
  "rob_ewma30|$P5|--mode realtime --prior-hours 720 --demean ewma --demean-halflife 720"
  "rob_tvint|$P5|--mode realtime --prior-hours 720 --intercept"
)

run_one() {
  IFS='|' read -r name panel opts <<< "$1"
  if [ -f "results/conn/$name/meta.json" ]; then echo "$name: exists"; return; fi
  # shellcheck disable=SC2086
  python ../code/run_connectedness.py --panel "$panel" $opts --name "$name" > "logs/$name.log" 2>&1
  echo "$name: $(grep -c seconds_total "logs/$name.log")/8 networks"
}
export -f run_one
export OMP_NUM_THREADS=3
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I{} bash -c 'run_one "$@"' _ {}
