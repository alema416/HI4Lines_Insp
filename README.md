# HI4Lines_Insp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


End-to-End MLOps Pipeline for Trustworthy Power-Lines-Inspection

![optim_diagram(16) drawio(1)](https://github.com/user-attachments/assets/65c0bc14-4d2a-466c-a593-b6ce7cafc6e1)

![image](https://github.com/user-attachments/assets/80c2dbd1-43da-4402-987e-30a17e18db5e)


Description: 

Please cite our journal paper as such:

Abstract:

## Project Organization

```
├── configs            <- Store project-wide editable variables and configurations.
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final data sets for modeling.
├── deploy             <- Source code and env for edge device inference.
├── docs               <- A default mkdocs project.
├── hailo_src          <- Source code for HAILO convertions.
├── hi4lines_insp      <- Source code for Data-Science part.
├── Makefile           <- Makefile with convenience commands for installation.
├── models             <- Placeholder for the model zoo.
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         '1.0-jqp-initial-data-exploration'.
├── pyproject.toml
├── README.md          <- The top-level README for developers using this project.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
├── st_src	       <- Source code for ST convertions.
└── requirements.txt   <- The requirements file for reproducing the data-science analysis environment.
```

--------

## Installation & Execution Instructions


### Data Science Server 

#### Step 1: Setup Data Science Module

Install virtualenv & download data via:

```
make create_environment
make requirements
make data
```

#### Step 2: Setup Quantizer/Emulator

* Install HAILO AI SW Suite via the instructions: 

https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2024-07/?sp_referrer=suite%2Fsuite_install.html#docker-installation

* with one change: in the .sh installer replace this: 

```
readonly SHARED_DIR=$repo_absolute_dir
```

(line 16)

And this: 

```
-v ${SHARED_DIR}/:/local/${SHARED_DIR}:rw \
```

(line 226)

Finally, from inside the docker container run: 
```
cd /local/<your_repo_path>/hailo_src
python3 -m pip install requirements.txt flask torchmetrics
sudo chmod -R a+w ..
python3 server.py
```

### Raspberry Pi

#### Setup Edge Device

Install virtualenv & download data via:

```
make create_environment && make requirements_rpi && make data
cd deploy
python3 run_classifier_optimization.py
```
