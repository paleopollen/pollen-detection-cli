# Pollen Detection Command Line Interface

This command-line program to detect pollen grains from images is developed based on pollen
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
python pollen_detection_cli.py
```

## Usage

```shell
sage: pollen_detection_cli.py [-h] --model-path [MODEL_FILE_PATH] --crops-dir [CROPS_DIR_PATH] [--detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX]] [--verbose]

Process PNG image stacks and detect pollen grains.

optional arguments:
  -h, --help            show this help message and exit
  --model-path [MODEL_FILE_PATH], -m [MODEL_FILE_PATH]
                        Full path of the trained model.
  --crops-dir [CROPS_DIR_PATH], -c [CROPS_DIR_PATH]
                        Full path of the directory containing the cropped image files.
  --detections-dir-prefix [DETECTIONS_DIR_PATH_PREFIX], -d [DETECTIONS_DIR_PATH_PREFIX]
                        Full path prefix of the directory to store the detection results.
  --verbose, -v         Display more details.
```
