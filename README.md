# HI4Lines_Insp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

description_too

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         hi4lines_insp and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── hi4lines_insp   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes hi4lines_insp a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# HI4Lines_Insp

Hierarchical Inference for Power Lines Inspection

![image](https://github.com/user-attachments/assets/80c2dbd1-43da-4402-987e-30a17e18db5e)


Description

Please cite our journal paper as such:

Citation

Abstract: Abstract

**TODO**

* differentiate *persample metrics* and *global metrics* (steamline pipeline infererence results)

## 1 - Setup Data Science Module

git clone repo
cd repo

Install virtualenv & download data via:

```
make
make 
```

## 2 - Setup Quantizer/Emulator

* git clone repo
* Install HAILO AI SW Suite via the instructions: https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2024-07/?sp_referrer=suite%2Fsuite_install.html#docker-installation
* and in the dockerfile replace this: readonly SHARED_DIR=<repo_dir>
and this                  -v ${SHARED_DIR}/:/local/${SHARED_DIR}:rw \
(lines 16 and 226)

flask
torchmetrics
hydra-core

cd hailo_src
sudo chmod -R a+w ..
ckpt2onnx and parser ok


## 3 - Edge Device Setup

Install virtualenv & download data via:

git clone repo
cd repo/deploy

Install virtualenv & download data via:

```
make
make 
```
