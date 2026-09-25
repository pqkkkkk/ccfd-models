"""Long Short-Term Memory (LSTM) Training Script for E-Commerce Fraud Detection.

This script implements Transaction-Sequence LSTM for fraud detection, grouping transactions
by user over time to learn behavioral patterns across sequential events. It evaluates
with comprehensive imbalanced fraud detection metrics (ROC-AUC, PR-AUC, F1-Score, G-Mean, etc.)
and saves trained models, preprocessors, and benchmark metrics matching the project standards.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


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

    # Priority pairs for sequence modeling: enriched files first (contains user_id and timestamp)
    candidate_pairs = [
        ("train_enriched.parquet", "test_enriched.parquet"),
        ("train_processed.parquet", "test_processed.parquet"),
        ("train.parquet", "test.parquet"),
        ("train_enriched.csv", "test_enriched.csv"),
        ("train_processed.csv", "test_processed.csv"),
        ("train.csv", "test.csv"),
    ]

    # 1. Direct check in input_path
    for tr_name, te_name in candidate_pairs:
        tr = input_path / tr_name
        te = input_path / te_name
        if tr.exists() and te.exists():
            return tr, te

    # 2. Search within subdirectories (up to depth 3)
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
        f"Looked for file pairs such as 'train_enriched.parquet' / 'test_enriched.parquet'."
    )


def detect_column(columns: List[str], candidates: List[str], requested: Optional[str] = None) -> Optional[str]:
    """Detect matching column from candidates or validate requested column."""
    if requested and requested.lower() != "auto":
        if requested in columns:
            return requested
        raise ValueError(f"Requested column '{requested}' not found. Available: {columns}")

    for c in candidates:
        if c in columns:
            return c
    return None


class SequencePreprocessor:
    """Preprocessor for tabular features in transaction sequence models."""

    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: List[str] = []

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        feature_names: List[str] = []

        if self.num_cols:
            num_data = df.select(self.num_cols).to_numpy().astype(np.float32)
            num_scaled = self.scaler.fit_transform(num_data)
            parts.append(num_scaled)
            feature_names.extend(self.num_cols)

        if self.cat_cols:
            cat_data = df.select(self.cat_cols).to_numpy().astype(str)
            cat_encoded = self.encoder.fit_transform(cat_data).astype(np.float32)
            parts.append(cat_encoded)
            encoded_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
            feature_names.extend(encoded_names)

        self.feature_names = feature_names
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []

        if self.num_cols:
            num_data = df.select(self.num_cols).to_numpy().astype(np.float32)
            num_scaled = self.scaler.transform(num_data)
            parts.append(num_scaled)

        if self.cat_cols:
            cat_data = df.select(self.cat_cols).to_numpy().astype(str)
            cat_encoded = self.encoder.transform(cat_data).astype(np.float32)
            parts.append(cat_encoded)

        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


class UserTransactionDataset(Dataset):
    """PyTorch Dataset organizing transactions into chronological user sequences."""

    def __init__(self, user_sequences: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.user_sequences = user_sequences

    def __len__(self) -> int:
        return len(self.user_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_sequences[idx]


def collate_user_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length user sequences into a single mini-batch tensor."""
    batch_x = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    lengths = torch.tensor([len(x) for x in batch_x], dtype=torch.long)

    padded_x = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    padded_y = pad_sequence(batch_y, batch_first=True, padding_value=0.0)

    # Mask: True for valid transaction timesteps, False for padded steps
    batch_size, max_len, _ = padded_x.shape
    idx_tensor = torch.arange(max_len).expand(batch_size, max_len)
    mask = idx_tensor < lengths.unsqueeze(1)

    return padded_x, padded_y, mask, lengths


class TransactionLSTM(nn.Module):
    """Long Short-Term Memory Network for User Transaction Sequence Fraud Detection."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        fc_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (B, L, in_features) -> Output: (B, L) logits."""
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits


class TabularLSTM(nn.Module):
    """Fallback Bidirectional LSTM for purely flat tabular data without user/time columns."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        fc_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features) -> (B, in_features, 1)
        x_seq = x.unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        last_step = out[:, -1, :]
        mean_pool = torch.mean(out, dim=1)
        combined = torch.cat([last_step, mean_pool], dim=-1)
        return self.classifier(combined).squeeze(-1)


def build_user_sequences(
    df: pl.DataFrame,
    features: np.ndarray,
    user_col: str,
    time_col: str,
    target_col: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Group features and labels into chronological sequences per user."""
    # Ensure dataframe is sorted by user and timestamp
    df_sorted = df.with_columns(pl.Series("__row_idx__", np.arange(len(df))))
    df_sorted = df_sorted.sort([user_col, time_col])

    user_groups = df_sorted.partition_by(user_col, as_dict=False)
    sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for group in user_groups:
        row_indices = group["__row_idx__"].to_numpy()
        x_user = torch.tensor(features[row_indices], dtype=torch.float32)
        y_user = torch.tensor(group[target_col].to_numpy().astype(np.float32), dtype=torch.float32)
        sequences.append((x_user, y_user))

    return sequences


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
        description="Train and evaluate Transaction-Sequence LSTM for E-Commerce Fraud Detection."
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
            "no_reduction_no_resampling",
        ),
        help="Path to input dataset directory (or folder containing enriched/train/test files).",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default=os.path.join("outputs", "05-09-2026", "lstm"),
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
        "--user-col",
        type=str,
        default="auto",
        help="User ID column name ('auto' detects 'user_id' or 'client_id', default: 'auto').",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="auto",
        help="Timestamp column name ('auto' detects 'timestamp' or 'transaction_time', default: 'auto').",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension size for LSTM layers (default: 64).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of stacked LSTM layers (default: 2).",
    )
    parser.add_argument(
        "--fc-dim",
        type=int,
        default=32,
        help="Dimension for MLP classification head (default: 32).",
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
        default=64,
        help="Mini-batch size in terms of users (default: 64).",
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
    print(" FRAUD DETECTION MODEL TRAINING: LONG SHORT-TERM MEMORY (LSTM)")
    print(f" Train File : {resolved_train}")
    print(f" Test File  : {resolved_test}")
    print(f" Output Dir : {output_dir}")
    device = select_device(args.device)
    print("=" * 70)

    print(">>> Loading dataset ...")
    def _read_df(path: Path) -> pl.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path)

    train_df = _read_df(resolved_train)
    test_df = _read_df(resolved_test)

    # Detect Target Column
    target_col = detect_column(train_df.columns, ["label", "is_fraud", "target", "fraud", "Class", "class"], args.target_col)
    if not target_col or target_col not in test_df.columns:
        raise ValueError(f"Could not resolve target column '{target_col}' in dataset.")

    # Detect User and Timestamp columns
    user_col = detect_column(train_df.columns, ["user_id", "client_id", "customer_id", "account_id"], args.user_col)
    time_col = detect_column(train_df.columns, ["timestamp", "transaction_time", "event_time", "time"], args.time_col)

    # Check if user/timestamp columns are available in a neighboring enriched file if current file is purely processed
    if not user_col or not time_col:
        neighbor_train = resolved_train.parent.parent / "feature_engineered" / "train_enriched.parquet"
        neighbor_test = resolved_test.parent.parent / "feature_engineered" / "test_enriched.parquet"
        if neighbor_train.exists() and neighbor_test.exists():
            print(f"    [Info] Found user sequence metadata in: {neighbor_train.parent}")
            en_train = pl.read_parquet(neighbor_train)
            en_test = pl.read_parquet(neighbor_test)
            if len(en_train) == len(train_df) and len(en_test) == len(test_df):
                train_df = train_df.with_columns([
                    en_train["user_id"].alias("user_id"),
                    en_train["timestamp"].alias("timestamp"),
                ])
                test_df = test_df.with_columns([
                    en_test["user_id"].alias("user_id"),
                    en_test["timestamp"].alias("timestamp"),
                ])
                user_col = "user_id"
                time_col = "timestamp"

    is_sequence_mode = (user_col is not None) and (time_col is not None)

    y_train = train_df[target_col].to_numpy().astype(int)
    y_test = test_df[target_col].to_numpy().astype(int)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    imbalance_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"    Target Column   : '{target_col}'")
    print(f"    Train shape     : {train_df.shape} (Neg: {n_neg:,}, Pos: {n_pos:,}, Imbalance ratio: {imbalance_ratio:.2f}:1)")
    print(f"    Test shape      : {test_df.shape}")
    print(f"    Execution Mode  : {'User Transaction Sequences' if is_sequence_mode else 'Tabular Feature-as-Sequence'}")

    # Determine numeric and categorical columns
    ignore_cols = {target_col, "event_id", "transaction_id"}
    if user_col:
        ignore_cols.add(user_col)
    if time_col:
        ignore_cols.add(time_col)

    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    cat_cols = [c for c in feature_cols if train_df[c].dtype in (pl.Utf8, pl.Categorical, pl.String)]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Preprocess features
    preprocessor = SequencePreprocessor(num_cols=num_cols, cat_cols=cat_cols)
    X_train_features = preprocessor.fit_transform(train_df)
    X_test_features = preprocessor.transform(test_df)
    in_features = X_train_features.shape[1]

    print(f"    Extracted {in_features} features ({len(num_cols)} numeric, {len(cat_cols)} categorical).")

    # Pos weight calculation
    if str(args.pos_weight).lower() == "auto":
        pos_weight_val = imbalance_ratio if imbalance_ratio > 2.0 else 1.0
    else:
        pos_weight_val = float(args.pos_weight)

    pos_weight_t = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    # Build DataLoader and Model
    if is_sequence_mode:
        print(f"    Grouping transactions into chronological user sequences by '{user_col}' ...")
        train_sequences = build_user_sequences(train_df, X_train_features, user_col, time_col, target_col)
        test_sequences = build_user_sequences(test_df, X_test_features, user_col, time_col, target_col)

        print(f"    Constructed {len(train_sequences):,} user sequences for training, {len(test_sequences):,} for testing.")

        train_dataset = UserTransactionDataset(train_sequences)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_user_sequences,
            drop_last=False,
        )

        test_dataset = UserTransactionDataset(test_sequences)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate_user_sequences,
            drop_last=False,
        )

        model = TransactionLSTM(
            in_features=in_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fc_dim=args.fc_dim,
            dropout=args.dropout,
        ).to(device)

    else:
        # Fallback Tabular mode
        print("    Running TabularLSTM on feature vectors ...")
        from torch.utils.data import TensorDataset

        X_train_t = torch.tensor(X_train_features, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_features, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=args.batch_size * 16,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t),
            batch_size=args.batch_size * 32,
            shuffle=False,
            drop_last=False,
        )

        model = TabularLSTM(
            in_features=in_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fc_dim=args.fc_dim,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(
        f"    Model architecture: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
        f"fc_dim={args.fc_dim}, dropout={args.dropout}, pos_weight={pos_weight_val:.2f}"
    )
    print(f"    Training for {args.epochs} epochs with batch_size={args.batch_size} ...")

    # Training Loop
    t_start_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        if is_sequence_mode:
            for batch_x, batch_y, mask, _ in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                # Compute loss only on valid non-padded transaction steps
                loss = criterion(logits[mask], batch_y[mask])
                loss.backward()
                optimizer.step()

                valid_count = mask.sum().item()
                total_loss += loss.item() * valid_count
                total_steps += valid_count
        else:
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_x)
                total_steps += len(batch_x)

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            print(f"      Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.5f}")

    train_time = time.perf_counter() - t_start_train

    # Evaluation Loop
    print("    Evaluating on test set ...")
    model.eval()
    t_start_eval = time.perf_counter()
    probs_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []

    with torch.no_grad():
        if is_sequence_mode:
            for batch_x, batch_y, mask, _ in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)

                mask_cpu = mask.cpu().numpy()
                probs_cpu = probs.cpu().numpy()
                batch_y_cpu = batch_y.cpu().numpy()

                probs_list.append(probs_cpu[mask_cpu])
                y_test_list.append(batch_y_cpu[mask_cpu])

            y_proba = np.concatenate(probs_list, axis=0)
            y_test_eval = np.concatenate(y_test_list, axis=0).astype(int)
        else:
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

            y_proba = np.concatenate(probs_list, axis=0)
            y_test_eval = y_test

    y_pred = (y_proba >= args.threshold).astype(int)
    eval_time = time.perf_counter() - t_start_eval

    metrics = evaluate_predictions(y_test_eval, y_pred, y_proba, train_time, eval_time)
    print_evaluation_report(metrics, title=f"EXPERIMENT: {output_dir.name}")

    # Save metrics JSON directly to output_dir
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "TransactionLSTM (PyTorch)" if is_sequence_mode else "TabularLSTM (PyTorch)",
                "train_file": str(resolved_train),
                "test_file": str(resolved_test),
                "target_col": target_col,
                "execution_mode": "user_transaction_sequence" if is_sequence_mode else "tabular_feature_sequence",
                "hyperparameters": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "fc_dim": args.fc_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "pos_weight": pos_weight_val,
                    "threshold": args.threshold,
                    "device": str(device),
                },
                "feature_count": in_features,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Save model weights & preprocessor directly to output_dir
    if not args.no_save_model:
        model_file = output_dir / "lstm_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "in_features": in_features,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "fc_dim": args.fc_dim,
                "dropout": args.dropout,
                "is_sequence_mode": is_sequence_mode,
            },
            model_file,
        )

        preprocessor_file = output_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_file)
        print(f"    Saved model to: {model_file}")
        print(f"    Saved preprocessor to: {preprocessor_file}")

    print(f"    Saved metrics to: {metrics_file}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
