# nemdetect-tubers

### Possible commands examples

Display settings:

``` sh
python3 main.py display-settings
```

Test data loading:

``` bash
python3 main.py test-load-data \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir \
```

Score model:

``` bash
python3 main.py score-model \
savgol-xgb \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir \
```

Optimize model:
``` bash
python3 main.py optimize-model \
savgol-xgb \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir \
```
