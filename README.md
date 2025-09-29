# An explainable machine learning approach for detection of potato cyst nematode infections using hyperspectral imaging  

See related [Publications](https://github.com/janezlapajne/manuscripts)

### 🔍 Introduction

Potato cyst nematodes pose a significant threat to potato cultivation. Early and accurate detection of infection is crucial for adequate management, as infestations often remain undetected for years and spread unnoticed. This emphasizes the need for reliable monitoring and early detection methods, which can cover large areas. Hyperspectral imaging shows great potential for non-invasive and early detection of nematode infections, but its ability to distinguish between biotic stressors (e.g. nematodes) and abiotic stressors (e.g. drought) still needs to be addressed. We investigated the stress responses of potato plants to two nematode species, Globodera rostochiensis and G. pallida, and water deficiency. Datasets were generated to isolate and evaluate the effects of single and combined stressors on plant physiology and morphology. Various machine learning methods and spectral processing techniques were applied to assess the distinctions between categories within the analyzed dataset. Exploratory methods were used to analyze the dataset and identify relevant spectral wavelengths, while statistical analyses assessed the significance of morphological and physiological parameters complementing the spectral data. The results show that water deficiency was the dominant factor in classification, reaching F1 values up to 0.95. The distinction between infected and non-infected plants reached a maximum F1 score of 0.70 in well-watered plants and 0.80 in water-deficient plants. Distinguishing nematode species and inoculation levels resulted in moderate F1 values (0.65–0.80), with performance improving to F1 = 0.80 when combining biotic and abiotic stress. However, accuracy decreased (F1 = 0.58) when classifying multiple stress categories simultaneously. These results highlight the challenges of separating stressors and the potential of hyperspectral imaging to detect nematode-infected plants.

**Authors:** Lapajne J., Susič N., Vončina A., Gerič Stare B., Viaene N., Van Beek J., Nuyttens D., Širca S., Žibrat U. \
**Keywords:** Hyperspectral imaging, machine learning, potato cyst nematodes, potato, Globodera, Solanum tuberosum, detection \
**Published In:** Plant Phenomics; SPJ \
**Publication Date:** Oct, 2025

---

### ⚙️ Environment setup

Setup tested on Ubuntu Linux machine with python 3.11.

It is recommended to use a Python environment manager, such as _PDM_, _Poetry_, _pipenv_, or other tools that support the use of `pyproject.toml` for project management.

To install PDM, create a virtual environment, and install dependencies, run script:

```bash
./scripts/install-dev.sh
```

---

### 🖼️ Dataset

Download the data from [Zenodo](https://zenodo.org/records/14267877) and unzip to folder named `data`.
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

```sh
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

-   Display settings

```sh
python3 main.py display-settings
```

-   Test data loading

```bash
python3 main.py test-load-data \
--group-id 0 \
--imaging-id 1 \
--imaging-id 2 \
--imaging-id 3 \
--camera-label vnir \
--camera-label swir
```

-   Score model without optimization

```bash
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

-   Optimize model

```bash
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

-   Generate only metrics

```bash
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

-   Generate only plots

```bash
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

-   Perform optimization and generate metrics, plots, and feature relevances.

```bash
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

```bash
export COMPUTE_RELEVANCES=True
```

---

### 📬 Contact

This project was initially developed by [Janez Lapajne](https://github.com/janezlapajne). If you have any questions or encounter any other problem, feel free to post an issue on Github.
