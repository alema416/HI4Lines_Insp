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

## Installation Instructions


### Data Science Server 

#### Step 1: Setup Data Science Module

Install virtualenv & download data via:

```
make create_environment && make requirements && make data
```

#### Step 2: Setup HAILO Quantizer/Emulator

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
python3 -m pip install requirements.txt
```

#### Step 3: Setup ST Quantizer/Emulator

### Raspberry Pi

#### Setup Edge Device

Install virtualenv & download data via:

```
make create_environment && conda activate HI4Lines_Insp && make requirements_rpi && make data
cd deploy/rpi/deploy
python3 run_classifier_optimization.py
```
### ST

#### Setup Edge Device

Install STM32 Model Zoo Services & add installation path to st_src corresponding config yaml.

## Execution Instructions

Step 1:

on Rpi activate the installed conda env & run:

```
tmux new-session -A -s server_rpi
cd <repo>/deploy/rpi/deploy/
python3 run_classifier_evaluator_server.py
```

Step 2:

on STM32MP257F-EV1 run:

Step 3:

on Data-Science Server activate the installed conda env & run:

./START_TRAINING.sh
on the bottom right pane ssh to the RPi and attach to the server_rpi session for monitoring.

#### Setup Edge Device


## Model Zoo 

https://drive.google.com/drive/folders/19vf1JSU2l2wGV99UFSCr1N1evp4B0wyJ?usp=sharing
