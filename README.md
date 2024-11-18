# The Belle II CDC AI Track (CAT) Finder

[![License: LGPL v3 or later](https://img.shields.io/badge/License-LGPL%20v3%20or%20later-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

This repository contains the training code used in the paper "End-to-End Multi-Track Reconstruction using Graph Neural Networks at Belle II". 

## Installation
This project requires **Python 3.9** or higher.
Clone this repository and setup the environment:
```
mkdir venv
python -m venv venv
source venv/bin/activate
source packages_install.sh
```

Install this package with the following command:
```
pip3 install . 
```

## Dataset
One example dataset is provided in [paper_dataset/cdchits](paper_dataset/cdchits), containing 10 events with a varying number of particles per event.

## Usage

For training, run  the following script:
```
python3 training/train_cat.py --config config/config.yaml --run pretraining 
```
For Inference, run the following script:
```
python3 training/evaluation.py --trainings_dir results/pretraining/
```
We do not provide trained models.

## Acknowledgments
It is a great pleasure to thank (in alphabetical order) Isabel Haide, Jan Kieseler, and Yannis Kluegl for discussions.
