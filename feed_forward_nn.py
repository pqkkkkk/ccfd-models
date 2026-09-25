"""Feed-Forward Neural Network (FFNN / MLP) Training Script for E-Commerce Fraud Detection.

This script loads preprocessed fraud detection dataset files, standardizes tabular features,
trains a Deep Feed-Forward Neural Network (PyTorch) with imbalanced loss handling,
evaluates it with comprehensive fraud detection metrics (ROC-AUC, PR-AUC, F1-Score,
G-Mean, etc.), and saves trained models, scalers, and benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def resolve_dataset_files(
    input_path: Path,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve train and test parquet/csv file paths from input directory or direct file paths."""
    if train_path and test_path:
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        return train_path, test_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        raise ValueError(
            f"Expected directory for --input, but got file: {input_path}. Use --train-path and --test-path."
        )

    candidate_pairs = [
        ("train_processed.parquet", "test_processed.parquet"),
        ("train.parquet", "test.parquet"),
        ("train_processed.csv", "test_processed.csv"),
        ("train.csv", "test.csv"),
    ]

    # 1. Direct check in input_path
    for tr_name, te_name in candidate_pairs:
        tr = input_path / tr_name
        te = input_path / te_name
        if tr.exists() and te.exists():
            return tr, te

    # 2. Search within subdirectories (e.g. artifacts/preprocessed) up to depth 3
    for root, _, _ in os.walk(input_path):
        root_path = Path(root)
        try:
            rel = root_path.relative_to(input_path)
            if len(rel.parts) > 3:
                continue
        except ValueError:
            continue

        for tr_name, te_name in candidate_pairs:
            tr = root_path / tr_name
            te = root_path / te_name
            if tr.exists() and te.exists():
                return tr, te

    raise FileNotFoundError(
        f"Could not find valid train/test data files in '{input_path}'. "
        f"Looked for file pairs such as 'train_processed.parquet' / 'test_processed.parquet'."
    )


def detect_target_col(columns: List[str], requested_col: Optional[str] = None) -> str:
    """Detect target column from dataset columns or validate user specified column."""
    if requested_col and requested_col.lower() != "auto":
        if requested_col in columns:
            return requested_col
        raise ValueError(
            f"Specified target column '{requested_col}' not found in dataset. Available: {columns}"
        )

    candidates = ["label", "is_fraud", "target", "fraud", "Class", "class"]
    for c in candidates:
        if c in columns:
            return c
    return columns[-1]


def load_dataset(
    train_path: Path, test_path: Path, target_col: str = "auto"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Load train and test datasets from parquet or csv files and return numpy arrays."""
    def _read_file(path: Path) -> pl.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(path)
        elif suffix == ".csv":
            return pl.read_csv(path)
        else:
            try:
                return pl.read_parquet(path)
            except Exception:
                return pl.read_csv(path)

    train_df = _read_file(train_path)
    test_df = _read_file(test_path)

    resolved_target = detect_target_col(train_df.columns, target_col)
    if resolved_target not in test_df.columns:
        raise ValueError(
            f"Target column '{resolved_target}' found in train set but missing in test set columns: {test_df.columns}"
        )

    feature_cols = [c for c in train_df.columns if c != resolved_target]

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_df[resolved_target].to_numpy().astype(int)

    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test = test_df[resolved_target].to_numpy().astype(int)

    return X_train, y_train, X_test, y_test, feature_cols, resolved_target


class TabularFFNN(nn.Module):
    """Deep Feed-Forward Neural Network for Tabular Fraud Detection."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


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
        "latency_per_sample_ms": round(latency_per_sample_ms, 4),
        "test_samples": int(len(y_true)),
        "test_fraud_samples": int(np.sum(y_true == 1)),
    }


def print_evaluation_report(metrics: Dict[str, Any], title: str = "EVALUATION REPORT") -> None:
    """Print formatted evaluation metrics report."""
    cm = metrics["confusion_matrix"]
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"  Test Samples       : {metrics['test_samples']} (Fraud: {metrics['test_fraud_samples']})")
    print(f"  Training Time      : {metrics['train_time_sec']:.2f} s")
    print(f"  Inference Time     : {metrics['eval_time_sec']:.3f} s ({metrics['latency_per_sample_ms']:.4f} ms/sample)")
    print("-" * 70)
    print(f"  ROC-AUC Score      : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC Score       : {metrics['pr_auc']:.4f}")
    print(f"  Fraud Precision    : {metrics['precision_fraud']:.4f}")
    print(f"  Fraud Recall       : {metrics['recall_fraud']:.4f}")
    print(f"  Fraud F1-Score     : {metrics['f1_fraud']:.4f}")
    print(f"  G-Mean             : {metrics['g_mean']:.4f}")
    print(f"  Specificity        : {metrics['specificity']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print("-" * 70)
    print(f"  Confusion Matrix   : TN={cm['tn']:,} | FP={cm['fp']:,} | FN={cm['fn']:,} | TP={cm['tp']:,}")
    print("=" * 70 + "\n")


def select_device(device_arg: str) -> torch.device:
    """Determine compute device (cuda / cpu)."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"    [Device] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("    [Device] GPU not available in PyTorch, using CPU.")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
        print(f"    [Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("    [Device] Using CPU.")
    return device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Feed-Forward Neural Network (FFNN / MLP) on a fraud detection dataset."
    )
    parser.add_argument(
        "--input",
        "--input-dir",
        "--base-dir",
        "-i",
        dest="input_path",
        type=str,
        default=os.path.join(
            "dataset",
            "e-commerce-fraud-detection",
            "preprocessed-05-09-2026",
            "has_resampling_dim_reduction",
        ),
        help="Path to input dataset directory (or folder containing artifacts/train/test files).",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default=os.path.join("outputs", "05-09-2026", "feed_forward_nn"),
        help="Directory to save models and evaluation reports.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Direct path to train parquet/csv file (overrides --input).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Direct path to test parquet/csv file (overrides --input).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="auto",
        help="Target column name in the dataset ('auto' detects 'label' or 'is_fraud', default: 'auto').",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="128,64,32",
        help="Comma-separated hidden layer dimensions (default: '128,64,32').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size (default: 1024).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for AdamW optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="L2 regularization weight decay (default: 0.0001).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2).",
    )
    parser.add_argument(
        "--pos-weight",
        type=str,
        default="auto",
        help="Positive class weight multiplier for BCEWithLogitsLoss ('auto' or float, default: 'auto').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target compute device: 'auto', 'cuda', or 'cpu' (default: 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Disable saving model weights and scalers to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path).resolve() if args.train_path else None
    test_path = Path(args.test_path).resolve() if args.test_path else None

    resolved_train, resolved_test = resolve_dataset_files(input_path, train_path, test_path)

    print("=" * 70)
    print(" FRAUD DETECTION MODEL TRAINING: FEED-FORWARD NEURAL NETWORK (FFNN)")
    print(f" Train File : {resolved_train}")
    print(f" Test File  : {resolved_test}")
    print(f" Output Dir : {output_dir}")
    device = select_device(args.device)
    print("=" * 70)

    print(">>> Loading dataset ...")
    X_train_raw, y_train, X_test_raw, y_test, feature_cols, target_col = load_dataset(
        resolved_train, resolved_test, target_col=args.target_col
    )

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    imbalance_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"    Target Column: '{target_col}'")
    print(f"    Train shape  : {X_train_raw.shape} (Neg: {n_neg:,}, Pos: {n_pos:,}, Imbalance ratio: {imbalance_ratio:.2f}:1)")
    print(f"    Test shape   : {X_test_raw.shape}")

    # Standardize features for Neural Network numerical stability
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Imbalance loss weighting
    if str(args.pos_weight).lower() == "auto":
        pos_weight_val = imbalance_ratio if imbalance_ratio > 2.0 else 1.0
    else:
        pos_weight_val = float(args.pos_weight)

    pos_weight_t = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    # Parse hidden dimensions
    hidden_dims = [int(h.strip()) for h in args.hidden_dims.split(",") if h.strip()]
    model = TabularFFNN(
        in_features=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    print(
        f"    Initializing TabularFFNN(in_dim={X_train.shape[1]}, hidden={hidden_dims}, dropout={args.dropout}, pos_weight={pos_weight_val:.2f}) ..."
    )
    print(f"    Training for {args.epochs} epochs with batch_size={args.batch_size} ...")

    # Training Loop
    t_start_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            print(f"      Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.5f}")

    train_time = time.perf_counter() - t_start_train

    print("    Evaluating on test set ...")
    model.eval()
    t_start_eval = time.perf_counter()
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=args.batch_size * 2, shuffle=False)
    probs_list = []

    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())

    y_proba = np.concatenate(probs_list, axis=0)
    y_pred = (y_proba >= args.threshold).astype(int)
    eval_time = time.perf_counter() - t_start_eval

    metrics = evaluate_predictions(y_test, y_pred, y_proba, train_time, eval_time)
    print_evaluation_report(metrics, title=f"EXPERIMENT: {output_dir.name}")

    # Save metrics JSON directly to output_dir
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "TabularFFNN (PyTorch)",
                "train_file": str(resolved_train),
                "test_file": str(resolved_test),
                "target_col": target_col,
                "hyperparameters": {
                    "hidden_dims": hidden_dims,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "pos_weight": pos_weight_val,
                    "threshold": args.threshold,
                    "device": str(device),
                },
                "feature_count": len(feature_cols),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Save model weights & scaler directly to output_dir
    if not args.no_save_model:
        model_file = output_dir / "ffnn_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "in_features": X_train.shape[1],
                "hidden_dims": hidden_dims,
                "dropout": args.dropout,
            },
            model_file,
        )

        scaler_file = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_file)
        print(f"    Saved model to: {model_file}")
        print(f"    Saved scaler to: {scaler_file}")

    print(f"    Saved metrics to: {metrics_file}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
