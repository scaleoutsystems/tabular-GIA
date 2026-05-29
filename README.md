# Profiling Privacy Preservation Under Gradient Inversion Attacks in Tabular Federated Learning

## Overview
This is the repository for the paper *Profiling Privacy Preservation Under Gradient Inversion Attacks in Tabular Federated Learning*.

## Installation
This section describes how to install the code and run the experiments used in the paper.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data
Datasets are not committed to this repository. Download the Adult or California Housing dataset.
```bash
cd tabular_gia
```
Adult:
```bash
chmod +x download_datasets.sh
./download_datasets.sh data
```
California Housing:
```bash
python data/download_california_housing.py
```

### Restricted data
MIMIC-IV requires credentialed PhysioNet access. Restricted datasets are not redistributed in this repository.

## Configuration
Make sure the config files in `configs/`:
- `base.py`
- `dataset/dataset.py`
- `fl/fedsgd.py`
- `fl/fedavg.py`
- `gia/gia.py`
- `model/model.py`

Make sure the configs are set to what you want to run and that for instance the data path correctly points to the downloaded dataset(s). The generic `sweep` experiment uses `configs/sweep.yaml`.

## Running
Run with the current config files:
```bash
python main.py
```
Outputs are written to `tabular_gia/results/`.


### Experiments
Values for `experiment_name` can be found in `tabular_gia/experiments/registry.py`. Configs are hardcoded in each experiment file.
```bash
python main.py --experiment [experiment_name]
```

for instance:
```bash
python main.py --experiment fedsgdbatchsizes
```
## Citation

Citation information will be added after publication.

## Authors
See the manuscript for full authorship.
