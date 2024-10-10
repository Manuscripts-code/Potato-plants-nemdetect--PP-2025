#!/usr/bin/env bash

set -e
set -x

group_ids=(0 1 18 19 21 22)
OPT="--do-optimize"
COM="run-all"

for group_id in "${group_ids[@]}"; do
  # pdm run main.py $COM savgol-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-pls-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-pls-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-ica-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-ica-xgb $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-kpca-svc $OPT --group-id "$group_id"
  # pdm run main.py $COM savgol-kpca-xgb $OPT --group-id "$group_id"
done

wait
