# The Belle II CDC AI Track (CAT) Finder

This repository contains the training code used in the paper "End-to-End Multi-Track Reconstruction using Graph Neural Networks at Belle II". 

## Dataset
One example dataset is provided in [paper_dataset/cdchits](paper_dataset/cdchits), containing 10 events with a varying number of particles per event.

## Installation
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

## Usage

For training, run  the following script:
```
python3 training/train_cat.py --config configs/config.yaml --run pretraining 
```
For Inference, run the following script:
```
python3 evaluation.py --trainings_dir results/pretraining/
```
We do not provide trained models.
