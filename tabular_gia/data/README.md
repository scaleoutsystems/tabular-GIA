---
license: cc-by-4.0
language:
- en
pretty_name: UCI Tabular Benchmark Sample
task_categories:
  - tabular-classification
  - tabular-regression
tags:
  - uci
  - tabular
  - classification
  - binary
  - multiclass
  - regression
size_categories:
  - 10K<n<100K
  - 100K<n<1M
---

# UCI Tabular Benchmark Sample
This repository contains a small collection of tabular datasets mirrored from the UCI Machine Learning Repository and prepared for convenient experimentation.

## Contents
The datasets are organized by common ML task type:

### Binary Classification
Adult dataset

Bank Marketing dataset

### Multiclassification
Covertype dataset

Statlog (Shuttle) dataset

### Regression
Year Prediction MSD dataset

## Source and License

All datasets in this repository are sourced from the UCI Machine Learning Repository and are used under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

- CC BY 4.0 license text: https://creativecommons.org/licenses/by/4.0/

### Upstream dataset pages (UCI)

- Adult (Census Income): https://archive.ics.uci.edu/dataset/2/adult
- Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Covertype: https://archive.ics.uci.edu/dataset/31/covertype
- Statlog (Shuttle): https://archive.ics.uci.edu/dataset/148/statlog+shuttle
- YearPredictionMSD: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd

## Modifications and Data Processing Notes
Some modifications may have been applied for usability and consistency. These modifications are intended to be non-substantive and not to change the meaning of the data.

Typical changes include:
- Adding **column headers** where the original files did not include headers.
- Converting original formats (for example space-separated or other delimiters) into **`.csv`**.
- Normalizing line endings and basic formatting fixes to improve parsing.
- In some cases, reorganizing files into a standard folder structure (for example `train.csv` and `test.csv`).

Unless explicitly stated in a dataset folder README (if present), no attempt was made to:
- Remove rows or features
- Alter feature values
- Rebalance classes
- Impute missing values

If you require a byte-for-byte identical copy of the upstream distribution, please download directly from the corresponding UCI page.

## Missing Values

Missing values are preserved as in the upstream sources. Depending on the dataset, missingness may appear as empty fields, `?`, or other dataset-specific markers. Refer to each dataset's UCI documentation for the authoritative description.

## Intended Use

This pack is intended for:
- Tabular model benchmarking (linear models, tree models, neural networks)
- Privacy and security research on tabular learning pipelines, including:
  - reconstruction and gradient inversion attacks
  - defenses such as clipping, noise injection, discretization, constraint-aware decoding
  - evaluating reconstructibility using feature-level and record-level metrics

It is not intended for making decisions about individuals or for any high-stakes deployment