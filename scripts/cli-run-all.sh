#!/usr/bin/env bash

set -e
set -x

group_ids=(0 1 18 19 21 22)
OPT="--do-optimize"

for group_id in "${group_ids[@]}"; do
  python3 main.py run-all savgol-xgb $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-svc $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-pls-svc $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-pls-xgb $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-ica-svc $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-ica-xgb $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-kpca-svc $OPT --group-id "$group_id" &
  python3 main.py run-all savgol-kpca-xgb $OPT --group-id "$group_id" &
done

wait
