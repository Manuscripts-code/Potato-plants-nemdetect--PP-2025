#!/usr/bin/env bash

set -e
set -x

# python3 main.py run-all savgol-xgb --do-optimize --group-id 0 &
# python3 main.py run-all savgol-svc --do-optimize --group-id 0 &
python3 main.py run-all savgol-pls-svc --do-optimize --group-id 0 &
# python3 main.py run-all savgol-pls-xgb --do-optimize --group-id 0 &
python3 main.py run-all savgol-ica-svc --do-optimize --group-id 0 &
# python3 main.py run-all savgol-ica-xgb --do-optimize --group-id 0 &
python3 main.py run-all savgol-kpca-svc --do-optimize --group-id 0 &
# python3 main.py run-all savgol-kpca-xgb --do-optimize --group-id 0 &

# python3 main.py run-all X --do-optimize --group-id 0

wait
