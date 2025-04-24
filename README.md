# HI4Lines_Insp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

End-to-End MLOps Pipeline for Trustworthy Power-Lines-Inspection

![image](https://github.com/user-attachments/assets/80c2dbd1-43da-4402-987e-30a17e18db5e)


Description

Please cite our journal paper as such:

Citation

Abstract: Abstract

## Project Organization

```
├── configs
├── data
├── deploy
├── docs
├── hailo_src
├── hi4lines_insp
├── Makefile
├── models
├── notebooks
├── pyproject.toml
├── README.md
├── references
├── reports
└── requirements.txt

```

--------

## Data Science Server 

### Step 1: Setup Data Science Module

Install virtualenv & download data via:

```
make create_environment
make requirements
make data
```

### Step 2: Setup Quantizer/Emulator

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

Finally: 
```
cd hailo_src
sudo chmod -R a+w ..
python3 server.py
```

## Raspberry Pi

### Setup Edge Device

Install virtualenv & download data via:

```
make create_environment && make requirements_rpi && make data
```