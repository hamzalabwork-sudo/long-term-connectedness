#!/usr/bin/env bash
# Download the R sources of ConnectednessApproach (and getPartition/getIndeces from
# frequencyConnectedness) from the CRAN mirror on GitHub, for validate_r.R.
# Usage (from work/):  bash ../code/fetch_rsrc.sh rsrc
set -eu
dir=${1:-rsrc}
mkdir -p "$dir"
ca=https://raw.githubusercontent.com/cran/ConnectednessApproach/master/R
for f in TVPVAR.R BayesPrior.R UninformativePrior.R IRF.R FEVD.R FrequencyConnectedness.R \
         TimeConnectedness.R ConnectednessTable.R ConnectednessApproach.R VFEVD.R; do
  curl -sfL "$ca/$f" -o "$dir/$f"
done
curl -sfL https://raw.githubusercontent.com/cran/frequencyConnectedness/master/R/tools.R -o "$dir/tools.R"
echo "fetched $(ls "$dir" | wc -l) files into $dir"
