"""
Feature engineering utilities for the ML Crystal Classifier project.

The input is expected to be an array of shape:
    [n_samples, 187, 8]

Each sample is a signal-like numerical array with 8 channels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def normalize_samples(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each sample independently to the range [0, 1].

    This removes absolute scale differences between samples while keeping
    the relative shape of each 187 x 8 signal array.

    Args:
        data: Array of shape [n_samples, timesteps, channels].
        eps: Small value to avoid division by zero.

    Returns:
        Normalized array with the same shape as data.
    """
    data = np.asarray(data, dtype=np.float64)
    sample_min = data.min(axis=(1, 2), keepdims=True)
    sample_max = data.max(axis=(1, 2), keepdims=True)
    sample_range = np.maximum(sample_max - sample_min, eps)
    return (data - sample_min) / sample_range


def safe_skewness_stat(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute a Pearson-style skewness statistic per channel.

    Formula:
        skewness ≈ 3 * (mean - median) / std
    """
    std = np.std(data, axis=1)
    return 3 * (np.mean(data, axis=1) - np.median(data, axis=1)) / np.maximum(std, eps)


def concatenate_features(feature_blocks: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate feature blocks along the feature dimension.

    Each block should have shape [n_samples, n_features].
    """
    return np.concatenate(feature_blocks, axis=1)


def extract_statistical_features(data: np.ndarray) -> np.ndarray:
    """
    Final feature extractor used for the cleaned GitHub version.

    This is based on the strongest feature-engineering attempt in the original
    notebook, which reached about 93.08% validation score in that setup.

    Args:
        data: Array of shape [n_samples, 187, 8].

    Returns:
        Feature matrix of shape [n_samples, n_engineered_features].
    """
    data = normalize_samples(data)

    std = np.std(data, axis=1)
    first_diff = np.diff(std, axis=1)
    second_diff = np.diff(first_diff, axis=1)
    skewness = safe_skewness_stat(data)
    max_values = np.max(data, axis=1)
    kurtosis = stats.kurtosis(data, axis=1, nan_policy="omit")

    feature_blocks = [
        std,
        first_diff,
        second_diff,
        skewness,
        max_values,
        kurtosis,
        stats.moment(data, 2, axis=1),
        stats.moment(data, 3, axis=1),
        stats.moment(data, 4, axis=1),
        stats.moment(data, 6, axis=1),
        stats.moment(data, 7, axis=1),
        stats.moment(data, 9, axis=1),
        stats.moment(data, 10, axis=1),
    ]

    features = concatenate_features(feature_blocks)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_hjorth_features(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Alternative signal-inspired feature set using activity, mobility, and complexity.

    This was an exploratory approach from the original notebook.
    """
    data = normalize_samples(data)

    dx = np.diff(data, axis=1)
    ddx = np.diff(dx, axis=1)

    activity = np.var(data, axis=1)
    var_dx = np.var(dx, axis=1)
    var_ddx = np.var(ddx, axis=1)

    mobility = np.sqrt(var_dx / np.maximum(activity, eps))
    complexity = np.sqrt(var_ddx / np.maximum(var_dx, eps)) / np.maximum(mobility, eps)

    skewness = safe_skewness_stat(data)
    max_values = np.max(data, axis=1)
    kurtosis = stats.kurtosis(data, axis=1, nan_policy="omit")

    feature_blocks = [
        activity,
        mobility,
        complexity,
        skewness,
        max_values,
        kurtosis,
        stats.moment(data, 2, axis=1),
        stats.moment(data, 3, axis=1),
        stats.moment(data, 4, axis=1),
        stats.moment(data, 6, axis=1),
        stats.moment(data, 7, axis=1),
        stats.moment(data, 9, axis=1),
        stats.moment(data, 10, axis=1),
    ]

    return np.nan_to_num(concatenate_features(feature_blocks), nan=0.0, posinf=0.0, neginf=0.0)
