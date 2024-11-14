#!/usr/bin/env bash

set -e
set -x

group_ids=($(seq 0 17))
OPT="--do-optimize"
# COM="train-model"
# COM="generate-metrics"
# COM="generate-plots"
# COM="calculate-relevances"
COM="run-all"

for group_id in "${group_ids[@]}"; do
  # pdm run main.py $COM savgol-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-pls-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-pls-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-ica-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-ica-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-pls-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-pls-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-ica-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM fft-ica-xgb $OPT --group-id "$group_id"

done

wait
