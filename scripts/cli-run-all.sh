#!/usr/bin/env bash

set -e
set -x

python3 main.py run-all savgol-xgb --do-optimize --group-id 0
python3 main.py run-all savgol-svc --do-optimize --group-id 0
python3 main.py run-all savgol-pls-svc --do-optimize --group-id 0
python3 main.py run-all savgol-pls-xgb --do-optimize --group-id 0
# python3 main.py run-all X --do-optimize --group-id 0
