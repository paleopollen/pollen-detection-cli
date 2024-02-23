# Pollen Detection Command Line Interface

This command-line software to detect pollen grains from images is developed based on pollen
detection [program](https://github.com/fengzard/ENSO_pollen_analysis/blob/main/03_Classification/03_00_Exporting_crops_for_Class.ipynb)
authored by [@fengzard](https://github.com/fengzard).

# Local Installation Instructions

Recommended Python version: 3.9

## Setup Virtual Environment

```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Command Line Interface

```shell
cd src
python pollen_detection_cli.py -m <model full path> -c <tile crops directory full path> -d <output detections directory>
```

## Usage

```shell
usage: pollen_detection_cli.py [-h] --model-path [MODEL_FILE_PATH] --crops-dir [CROPS_DIR_PATH] [--detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX]] [--num-processes [NUM_PROCESSES]] [--batch-size [BATCH_SIZE]] [--verbose]

Process PNG image stacks and detect pollen grains.

optional arguments:
  -h, --help            show this help message and exit
  --model-path [MODEL_FILE_PATH], -m [MODEL_FILE_PATH]
                        Full path of the trained model.
  --crops-dir [CROPS_DIR_PATH], -c [CROPS_DIR_PATH]
                        Full path of the directory containing the cropped image files.
  --detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX], -d [DETECTIONS_DIR_PATH_PREFIX]
                        Full path prefix of the directory to store the detection results.
  --num-processes [NUM_PROCESSES], -n [NUM_PROCESSES]
                        Number of processes to use for parallel processing.
  --batch-size [BATCH_SIZE], -b [BATCH_SIZE]
                        Batch size for parallel processing.
  --verbose, -v         Display more details.
```

## Docker Installation Instructions

### Build Docker Image

```shell
docker build -t pollen-detection .
```

### Run Command Line Interface

```shell
docker run -it --rm -v $(pwd)/data:/data --name pollen-detection-container pollen-detection -m /data/model.h5 -c /data/crops -d /data/detections
```
