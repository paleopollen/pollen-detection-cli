FROM python:3.9-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
&& mkdir -p /data \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ /usr/src/app/

ENTRYPOINT [ "python3", "pollen_detection_cli.py"]