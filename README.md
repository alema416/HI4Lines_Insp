# HI4Lines_Insp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


End-to-End MLOps Pipeline for Trustworthy Power-Lines-Inspection

![optim_diagram(16) drawio(2)](https://github.com/user-attachments/assets/1799e476-9775-4202-a6e8-657846810fa5)

![image](https://github.com/user-attachments/assets/80c2dbd1-43da-4402-987e-30a17e18db5e)


Description: 

Please cite our journal paper as such:

Abstract:

## Project Organization

```
├── L-ML               <- Source code for L-ML part.
├── configs            <- Store project-wide editable variables and configurations.
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final data sets for modeling.
├── deploy             <- Source code and env for edge device inference.
├── docs               <- A default mkdocs project.
├── hailo_src          <- Source code for HAILO convertions.
├── hi4lines_insp      <- Source code for Data-Science part.
├── models             <- Placeholder for the model zoo.
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         '1.0-jqp-initial-data-exploration'.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
├── st_src             <- Source code for ST convertions.
├── .gitignore
├── Makefile           <- Makefile with convenience commands for installation.
├── README.md          <- The top-level README for developers using this project.
├── pyproject.toml
└── requirements.txt   <- The requirements file for reproducing the data-science analysis environment.
```

--------

# Installation

## Step 1: Install common

```
docker compose pull tensorboard dashboard postgres 
docker compose build train 
```

## Step 2: Install Hardware-Specific components

### For DeGirum Orca / Google Coral

```
docker compose pull compiler_api
```

### For Hailo

manually download hailo_ai_suite.zip from https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2024-07/?sp_referrer=suite%2Fsuite_install.html#docker-installation and run
https://hailo.ai/developer-zone/sw-downloads/

```
docker load --input hailo_ai_sw_suite_2025-04.tar.gz
docker compose build hailo
```

### For Stm32 Devices

```
docker compose build stm32
```

# Execution

Step 1: edit the configs to match your requirements

Step 2: initialize backend and start the optimization process 

```
docker compose up -d postgres minio
docker compose run -d --rm train
docker compose up -d dashboard tensorboard
```

Step 3: Depending on target hardware

```
docker compose up -d < compiler_api | hailo | stm32ai > 
```

# Monitoring

You can monitor the progress of the optimization in real-time on localhost:6007, details for each trial on localhost:8080 and the artifacts are on localhost:9001.

# Termination

All the running processes are killed via:

```
docker compose down
```
