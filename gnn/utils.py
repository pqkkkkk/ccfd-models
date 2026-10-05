"""Utility functions for GNN experiments in Fraud Detection.

Includes evaluation metrics calculation and pretty reporting, consistent with
Random Forest and XGBoost baselines in the repository.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    train_time: float,
    eval_time: float,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for fraud detection."""
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    accuracy = float(accuracy_score(y_true, y_pred))

    # Binary metrics for fraud (class 1)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Macro & Weighted metrics
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix & Specificity / G-Mean
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        g_mean = float(np.sqrt(recall * specificity))
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0
        g_mean = 0.0

    latency_per_sample_ms = (eval_time / len(y_true)) * 1000.0 if len(y_true) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "f1_fraud": round(f1, 5),
        "precision_fraud": round(precision, 5),
        "recall_fraud": round(recall, 5),
        "specificity": round(specificity, 5),
        "g_mean": round(g_mean, 5),
        "accuracy": round(accuracy, 5),
        "f1_macro": round(f1_macro, 5),
        "f1_weighted": round(f1_weighted, 5),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "train_time_sec": round(train_time, 3),
        "eval_time_sec": round(eval_time, 3),
        "latency_per_sample_ms": round(latency_per_sample_ms, 5),
        "test_samples": int(len(y_true)),
        "test_fraud_samples": int(np.sum(y_true == 1)),
    }


def print_evaluation_report(metrics: Dict[str, Any], title: str = "EVALUATION REPORT") -> None:
    """Print formatted evaluation metrics report."""
    cm = metrics["confusion_matrix"]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f" {title}")
    print(sep)
    print(f"  ROC-AUC          : {metrics['roc_auc']:.5f}")
    print(f"  PR-AUC           : {metrics['pr_auc']:.5f}")
    print(f"  F1-Score (Fraud) : {metrics['f1_fraud']:.5f}")
    print(f"  Precision (Fraud): {metrics['precision_fraud']:.5f}")
    print(f"  Recall (Fraud)   : {metrics['recall_fraud']:.5f}")
    print(f"  Specificity      : {metrics['specificity']:.5f}")
    print(f"  G-Mean           : {metrics['g_mean']:.5f}")
    print(f"  Accuracy         : {metrics['accuracy']:.5f}")
    print(f"  F1 (Macro)       : {metrics['f1_macro']:.5f}")
    print(f"  F1 (Weighted)    : {metrics['f1_weighted']:.5f}")
    print(f"  Confusion Matrix : TP={cm['tp']:,} | FP={cm['fp']:,} | FN={cm['fn']:,} | TN={cm['tn']:,}")
    print(f"  Train Time       : {metrics['train_time_sec']:.2f}s")
    print(f"  Eval Time        : {metrics['eval_time_sec']:.2f}s ({metrics['latency_per_sample_ms']:.4f} ms/sample)")
    print(f"{sep}\n")


def get_device(cuda_device: Optional[int] = 0) -> torch.device:
    """Return torch.device (CUDA if available and requested, else CPU)."""
    if torch.cuda.is_available() and cuda_device is not None:
        return torch.device(f"cuda:{cuda_device}")
    return torch.device("cpu")
