# ML Crystal Classifier

Feature-engineering project for a binary classification challenge based on synthetic signal arrays.

Each sample is a `187 × 8` numerical "pulse" array. The goal is to classify each sample as one of two classes, originally framed as **Ruby** vs **Sapphire**, using a constrained Decision Tree model. Instead of using a large neural network, this project focuses on designing compact statistical features that make a shallow/interpretable classifier perform well.

## Project Summary

- Built a feature-engineering pipeline for high-dimensional signal-like arrays.
- Normalized each sample independently to remove absolute scale differences.
- Extracted statistical descriptors such as standard deviation, skewness, kurtosis, higher-order moments, max values, and finite-difference features.
- Evaluated features with a depth-limited `DecisionTreeClassifier`.
- Reached about **93.08% validation accuracy / ROC-AUC-style score** in the original notebook setup.

## Why this project matters

The challenge was not just to fit a model. The model class was intentionally constrained, so performance depended heavily on feature design. This makes the project closer to practical ML work where representation quality, signal preprocessing, and metric-driven iteration matter as much as model selection.

## Repository Structure

```text
ml-crystal-classifier/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ml_crystal_classifier_clean.ipynb
└── src/
    ├── __init__.py
    ├── features.py
    ├── evaluate.py
    └── visualize.py
```

## Methods

### Normalization

Each sample is normalized independently:

```text
x_norm = (x - min(x)) / (max(x) - min(x))
```

This preserves each signal's relative shape while reducing the effect of absolute magnitude differences.

### Feature engineering

The final feature set includes:

- per-channel standard deviation
- first and second differences of standard deviation
- skewness-like asymmetry statistic
- max values
- kurtosis
- higher-order statistical moments

### Classifier

The challenge uses Decision Tree classifiers with limited depth, so the feature extractor is designed to summarize the raw `187 × 8` signal into a compact tabular representation.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the clean notebook:

```bash
jupyter notebook notebooks/ml_crystal_classifier_clean.ipynb
```

Or import the feature function in Python:

```python
from src.features import extract_statistical_features
```

## Notes

The original notebook was developed during IOAI Canada training. This repository is a cleaned version meant to show the final approach, while preserving the core human-written feature-engineering logic.
