# nemdetect-tubers

See related [Publications](https://github.com/janezlapajne/manuscripts)

### 🔍 Introduction

Place abstract here.

**Authors:**  \
**Keywords:**  \
**Published In:**  \
**Publication Date:**

---

### ⚙️ Environment setup

Setup tested on Ubuntu Linux machine with python 3.11.

It is recommended to use a Python environment manager, such as *PDM*, *Poetry*, *pipenv*, or other tools that support the use of `pyproject.toml` for project management.

To install PDM, create a virtual environment, and install dependencies, run script:

```bash
./scripts/install-dev.sh
```

---

### 🖼️ Dataset

Download the data from [Zenodo](ADD-LINK) and unzip to folder named `data`.
The folder structure should look like:

```
📂 data
├── 📁 imaging1
│   ├── 📄labels.csv
│   ├── 📄signatures_swir.csv
│   └── 📄signatures_vnir.csv
├── 📁 imaging2
│   └── 📄 ...
├── 📁 imaging3
│    └── 📄 ...
└── 📐 bands.json
```

---

### 📚 How to use

The repo is setup as cli interface so the execution of commands is made easier. To see available options run:

``` sh
python3 main.py --help
```

Which will display:

```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion          Install completion for the current shell.
  --show-completion             Show completion for the current shell, to copy it or customize the installation.
  --help                        Show this message and exit.

Commands:
  calculate-relevances
  display-metrics
  display-settings
  generate-metrics
  generate-plots
  run-all
  test-load-data
  train-model
```

**Examples**

The presented examples are written for the `savgol-xgb` model, but can be used with any other model.

- Display settings

``` sh
python3 main.py display-settings
```

- Test data loading

``` bash
python3 main.py test-load-data \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

- Score model without optimization

``` bash
python3 main.py train-model \
savgol-xgb \
--no-do-optimize \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

- Optimize model

``` bash
python3 main.py train-model \
savgol-xgb \
--do-optimize \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

- Generate only metrics

``` bash
python3 main.py generate-metrics \
savgol-xgb \
--do-optimize \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

- Generate only plots

``` bash
python3 main.py generate-plots \
savgol-xgb \
--do-optimize \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

- Perform optimization and generate metrics, plots, and feature relevances.

``` bash
python3 main.py run-all \
savgol-xgb \
--do-optimize \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

⚠️ **Note**

To enable relevance calculations and generate corresponding plots, ensure you configure your environment by setting the required variable. Add the following line to your .env file or execute it in your terminal:

``` bash
export COMPUTE_RELEVANCES=True
```

---

### 📬 Contact

This project was initially developed by [Janez Lapajne](https://github.com/janezlapajne). If you have any questions or encounter any other problem, feel free to post an issue on Github.
