# A Command Line Interface for Pollen Detection on Z-Stack Images

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14497719.svg)](https://doi.org/10.5281/zenodo.14497719)
[![Docker](https://github.com/paleopollen/pollen-detection-cli/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/paleopollen/pollen-detection-cli/actions/workflows/docker-publish.yml)

We present, `Pollen Detector` a Command Line Interface (CLI) for pollen detection on Z-stack images. It is developed
based on Deep learning and Pollen Detection in the Open
World [notebooks](https://github.com/fengzard/open_world_pollen_detection) authored by Jennifer T. Feng, Shu Kong, Timme
H. Donders, and Surangi W. Punyasena.

## Docker Installation Instructions (Recommended)

### Build Docker Image

```shell
docker build -t pollen-detection .
```

### Run Command Line Interface

#### Example serial mode command

```shell
docker run -it --rm -v $(pwd)/data:/data --name pollen-detection-container pollen-detection -m /data/models/model_1.h5 -c /data/crops -d /data/detections
```

#### Example parallel mode command

```shell
docker run -it --shm-size=<memory_size_allocated> --rm -v $(pwd)/data:/data --name pollen-detection-container pollen-detection -m /data/models/model_1.h5 -c /data/crops -d /data/detections -p -n 4 -b 4
```

**Important:** Here, we use the `--shm-size=<memory_size_allocated>` option to increase the shared memory size for the
Docker Engine. The default value is 64MB, which may be too little for parallel processing. Set this value according to
the memory available on the host machine container.

Help command:

```shell
docker run -it --rm pollen-detection --help
```

## Local Installation Instructions (Alternative)

Recommended Python version: 3.9

### Setup Virtual Environment

```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Command Line Interface

#### Example serial mode command

```shell
cd src
python pollen_detection_cli.py -m <model file full path> -c <tile crops directory full path> -d <output detections directory full path prefix>
```

#### Example parallel mode command

```shell
cd src
python pollen_detection_cli.py -m <model file full path> -c <tile crops directory full path> -d <output detections directory full path prefix> -p -n 4 -b 4
```

### Usage

```shell
usage: python pollen_detection_cli.py [-h] --model-path [MODEL_FILE_PATH] --crops-dir [CROPS_DIR_PATH] 
                               [--detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX]] [--parallel] 
                               [--num-processes [NUM_PROCESSES]] [--num-workers [NUM_WORKERS]]
                               [--batch-size [BATCH_SIZE]] [--shuffle] [--cpu] [--verbose]

Process PNG image stacks and detect pollen grains.

optional arguments:
  -h, --help            show this help message and exit
  --model-path [MODEL_FILE_PATH], -m [MODEL_FILE_PATH]
                        Full path of the trained model.
  --crops-dir [CROPS_DIR_PATH], -c [CROPS_DIR_PATH]
                        Full path of the directory containing the cropped image files.
  --detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX], -d [DETECTIONS_DIR_PATH_PREFIX]
                        Full path prefix of the directory to store the detection results.
  --parallel, -p        Run the detection in parallel.
  --num-processes [NUM_PROCESSES], -n [NUM_PROCESSES]
                        Number of processes to use for parallel processing.
  --num-workers [NUM_WORKERS], -w [NUM_WORKERS]
                        Number of data loading workers to use for parallel processing.
  --batch-size [BATCH_SIZE], -b [BATCH_SIZE]
                        Batch size for parallel processing.
  --shuffle, -s         Shuffle the dataset.
  --cpu                 Run the detection on CPU only.
  --verbose, -v         Display more details.
```
