# Pollen Detection Command Line Interface

This command-line program to detection pollen grains from images is developed based on pollen
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
usage: pollen_detection_cli.py [-h] --model-dir [MODEL_DIR_PATH] --crops-dir [CROPS_DIR_PATH] [--detections-dir [CROPS_DIR_PATH]] [--verbose]

Process PNG image stacks and detect pollen grains.

optional arguments:
  -h, --help            show this help message and exit
  --model-dir [MODEL_DIR_PATH], -m [MODEL_DIR_PATH]
                        Full path of the directory containing the model files.
  --crops-dir [CROPS_DIR_PATH], -c [CROPS_DIR_PATH]
                        Full path of the directory containing the cropped image files.
  --detections-dir [CROPS_DIR_PATH], -d [CROPS_DIR_PATH]
                        Full path of the directory to store the detection results.
  --verbose, -v         Display more details.

```

