#!/usr/bin/env bash
#
# download_hailo.sh
# Script to download the Hailo AI SW Suite 2025-04 Docker ZIP from CloudFront

URL="https://d1rcmhofac3v9q.cloudfront.net/DevZone-Data/SW%20Downloads/Hailo%20SW%20Suite/2025-04-01/hailo_ai_sw_suite_2025-04_docker.zip?Expires=1750837833&Signature=X7gjKH-iLobdD32hIE88yKpcQc3wesGRfsfAzm3moT9aJNnHme80mQI-LqZsivqfUYrkkx56QlBaFtVrBfo8ldU8xnskoQ0d~PiPM2Xzzrj72chhnxDyCJPhsAHDvP1jCGDvrbdrQLyW-RWkjYk2dOuHox3y3s1EhLJJKOCpvokvRUXxsBpXMF7SOx7-QN2wsoRmm96PLXuNJ~y5aCC8~EDv3~hVzwkbySUdXsr26QttoNPoghl5P5O66y-x2hMeL4T7NGIExnnZ~WqQ__&Key-Pair-Id=K3T77PKOVL0N77"
OUTPUT="hailo_ai_sw_suite_2025-04_docker_1.zip"

echo "Downloading ${OUTPUT} from CloudFront..."
curl -L -o "${OUTPUT}" "${URL}"

if [ $? -eq 0 ]; then
  echo "Download complete: ${OUTPUT}"
else
  echo "Download failed." >&2
  exit 1
fi
