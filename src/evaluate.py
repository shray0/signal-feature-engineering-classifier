"""
Evaluation helpers for the ML Crystal Classifier project.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(
    train_features: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 20,
    random_state: int = 2024,
) -> DecisionTreeClassifier:
    """Train a depth-limited Decision Tree classifier."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(train_features, y_train)
    return model


def evaluate_classifier(
    train_features: np.ndarray,
    val_features: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    max_depth: int = 20,
) -> dict[str, float]:
    """
    Train a Decision Tree and evaluate accuracy + ROC-AUC.

    The original challenge used ROC-AUC-style scoring, while the resume/GitHub
    summary often refers to classification accuracy. This function reports both.
    """
    model = train_decision_tree(train_features, y_train, max_depth=max_depth)
    preds = model.predict(val_features)

    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "roc_auc": float(roc_auc_score(y_val, preds)),
    }
