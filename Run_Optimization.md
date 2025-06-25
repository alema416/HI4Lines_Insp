# Run Optimization

## Installation and Execution

```
docker-compose build train

docker-compose up -d postgres

docker-compose run -d --rm train

docker-compose up -d dashboard tensorboard
```

### for ORCA/CORAL:

```
docker-compose up -d compiler_api
```

### for ORCA/CORAL:

```
docker-compose up -d stm32_services
```

### for HAILO:
```
curl -L -o hailo_ai_sw_suite_2025-04_docker.zip "https://d1rcmhofac3v9q.cloudfront.net/DevZone-Data/SW%20Downloads/Hailo%20SW%20Suite/2025-04-01/hailo_ai_sw_suite_2025-04_docker.zip?Expires=1750837833&Signature=X7gjKH-iLobdD32hIE88yKpcQc3wesGRfsfAzm3moT9aJNnHme80mQI-LqZsivqfUYrkkx56QlBaFtVrBfo8ldU8xnskoQ0d~PiPM2Xzzrj72chhnxDyCJPhsAHDvP1jCGDvrbdrQLyW-RWkjYk2dOuHox3y3s1EhLJJKOCpvokvRUXxsBpXMF7SOx7-QN2wsoRmm96PLXuNJ~y5aCC8~EDv3~hVzwkbySUdXsr26QttoNPoghl5LqurIKLnAUOXH1zTXldVmhxSn8b9Iz9uJjxqjVld~WSp7jhy4ernH8TPwys5n1RNi0FRi2P5O66y-x2hMeL4T7NGIExnnZ~WqQ__&Key-Pair-Id=K3T77PKOVL0N77"
unzip hailo_ai_sw_suite_2025-04_docker.zip
```
(it contains one .sh file and one .tar.gz. file)

edit these lines of .sh file to match the directory of the repo in your system:

line 16: readonly SHARED_DIR=<repo absolute path in your system>
line 226: -v ${SHARED_DIR}/:/local/HI4Lines_Insp:rw \
line 260: replace the whole function run_hailo_ai_sw_suite_image with:

```

```

then to install:
```
./hailo_ai_sw_suite_docker_run.sh
```
and type "exit".

and the next times:
```
./hailo_ai_sw_suite_docker_run.sh --resume
```


to kill the whole pipeline:

```
docker-compose down -v
```
