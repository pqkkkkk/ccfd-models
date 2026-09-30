"""Convolutional Neural Network (CNN) Training Script for E-Commerce Fraud Detection.

This script implements the 2D Feature Matrix CNN architecture for fraud detection as
described in CNN.pdf (based on Yann LeCun 1998 & Zheng et al. 'Credit Card Fraud Detection
using Convolutional Neural Networks').

Core Mechanism:
1. Feature Transformation:
   - Tabular transaction events are transformed into 2D Feature Matrices (H x W) based on
     the principle of Local Correlation.
   - Rows (Feature Types): Behavioral metrics including AvgAmount, TotalAmount, BiasAmount,
     Transaction Count, Merchant Diversity, Country Diversity, Channel Diversity, and Amount Std.
   - Columns (Time Windows): Multi-scale past observation windows (e.g. 1d, 2d, 3d, 7d, 14d,
     30d, 60d, 90d).
2. Deep 2D-CNN Architecture:
   - Convolutional layers with 3x3 kernels slide across both the feature dimension and the
     temporal dimension to capture latent spatio-temporal transaction patterns.
   - Max Pooling, ReLU activations, and Dropout prevent overfitting.
   - Optional Dual-Branch fusion combines the 2D behavioral matrix with instantaneous
     transaction context features.
3. Imbalanced Learning & Standardized Evaluation:
   - Handles severe class imbalance via BCEWithLogitsLoss with positive class weighting.
   - Comprehensive metrics: ROC-AUC, PR-AUC, F1-Score, Recall, Precision, G-Mean, Specificity,
     Confusion Matrix, and Inference Latency.
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
from torch.utils.data import DataLoader, TensorDataset


FEATURE_TYPES = [
    "avg_amount",      # Average transaction amount in window
    "total_amount",    # Cumulative transaction amount in window
    "bias_amount",     # Difference: current amount - window avg amount
    "count",           # Number of transactions in window
    "uniq_merch",      # Number of distinct merchant categories in window
    "uniq_country",    # Number of distinct transaction countries in window
    "uniq_channel",    # Number of distinct transaction channels in window
    "std_amount",      # Standard deviation of amounts in window
]

DEFAULT_WINDOWS = ["1d", "2d", "3d", "7d", "14d", "30d", "60d", "90d"]


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

    # Priority candidate pairs: enriched files with timestamp and user_id are preferred
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

    # 2. Search within subdirectories up to depth 4
    for root, _, _ in os.walk(input_path):
        root_path = Path(root)
        try:
            rel = root_path.relative_to(input_path)
            if len(rel.parts) > 4:
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


def construct_2d_feature_matrices(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    user_col: str,
    time_col: str,
    amount_col: str,
    merch_col: Optional[str] = None,
    country_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    windows: List[str] = DEFAULT_WINDOWS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Transform transaction sequence into 2D Feature Matrices (H x W).

    Rows (H = 8): Feature Types organized by Local Correlation.
    Cols (W = 8): Time Windows in ascending chronological horizon.
    """
    print(f"    Constructing 2D Feature Matrices using {len(FEATURE_TYPES)} Feature Types x {len(windows)} Windows ...")
    t0 = time.time()

    # Track split origin
    tr_len = len(train_df)
    te_len = len(test_df)
    train_tagged = train_df.with_columns(pl.lit(1).alias("__is_train"), pl.int_range(0, tr_len).alias("__orig_idx"))
    test_tagged = test_df.with_columns(pl.lit(0).alias("__is_train"), pl.int_range(0, te_len).alias("__orig_idx"))

    # Concat and sort chronologically by user
    combined = pl.concat([train_tagged, test_tagged]).sort([user_col, time_col])

    # Build rolling statistics for each window
    matrix_feature_names: List[str] = []
    rolled_expressions: List[pl.Expr] = []

    for w in windows:
        rolled = combined.rolling(
            index_column=time_col,
            period=w,
            group_by=user_col,
            closed="left",  # strictly causal: only past transactions before current timestamp
        ).agg([
            pl.col(amount_col).mean().alias(f"avg_amount_{w}"),
            pl.col(amount_col).sum().alias(f"total_amount_{w}"),
            pl.len().alias(f"count_{w}"),
            (
                pl.col(merch_col).n_unique().alias(f"uniq_merch_{w}")
                if merch_col and merch_col in combined.columns
                else pl.lit(0.0).alias(f"uniq_merch_{w}")
            ),
            (
                pl.col(country_col).n_unique().alias(f"uniq_country_{w}")
                if country_col and country_col in combined.columns
                else pl.lit(0.0).alias(f"uniq_country_{w}")
            ),
            (
                pl.col(channel_col).n_unique().alias(f"uniq_channel_{w}")
                if channel_col and channel_col in combined.columns
                else pl.lit(0.0).alias(f"uniq_channel_{w}")
            ),
            pl.col(amount_col).std().alias(f"std_amount_{w}"),
        ])

        for ft in ["avg_amount", "total_amount", "count", "uniq_merch", "uniq_country", "uniq_channel", "std_amount"]:
            col_name = f"{ft}_{w}"
            combined = combined.with_columns(rolled[col_name].fill_null(0.0).alias(col_name))

        # Bias amount: current amount - past average amount
        combined = combined.with_columns(
            (pl.col(amount_col) - pl.col(f"avg_amount_{w}")).alias(f"bias_amount_{w}")
        )

    # Order columns strictly by (Feature Types x Windows) for local correlation grid
    for ft in FEATURE_TYPES:
        for w in windows:
            matrix_feature_names.append(f"{ft}_{w}")

    # Separate train and test restoring original order
    train_res = combined.filter(pl.col("__is_train") == 1).sort("__orig_idx")
    test_res = combined.filter(pl.col("__is_train") == 0).sort("__orig_idx")

    X_train_raw = train_res.select(matrix_feature_names).to_numpy().astype(np.float32)
    X_test_raw = test_res.select(matrix_feature_names).to_numpy().astype(np.float32)

    # Standardize features row-wise across windows
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Reshape to (N, 1, H, W) where H = len(FEATURE_TYPES), W = len(windows)
    H = len(FEATURE_TYPES)
    W = len(windows)
    X_train_matrix = X_train_scaled.reshape(-1, 1, H, W)
    X_test_matrix = X_test_scaled.reshape(-1, 1, H, W)

    elapsed = time.time() - t0
    print(f"    Feature matrices built successfully in {elapsed:.2f}s | Shape: Train {X_train_matrix.shape}, Test {X_test_matrix.shape}")
    return X_train_matrix, X_test_matrix, matrix_feature_names


class TabularContextPreprocessor:
    """Preprocessor for instant transaction context features (Wide branch)."""

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
            feature_names.extend(self.encoder.get_feature_names_out(self.cat_cols).tolist())

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


class FraudCNN2D(nn.Module):
    """Deep 2D Convolutional Neural Network for Fraud Detection with Optional Tabular Branch.

    Following the architecture:
    - Input: 2D Feature Matrix of shape (Batch, 1, H, W)
    - Conv Block 1: 3x3 Conv -> BatchNorm2d -> ReLU
    - Conv Block 2: 3x3 Conv -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    - Conv Block 3: 3x3 Conv -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((2, 2))
    - Flatten & Dense Projection
    - Optional Wide Branch for instantaneous tabular attributes
    - Fully Connected Classification Head
    """

    def __init__(
        self,
        in_channels: int = 1,
        grid_height: int = 8,
        grid_width: int = 8,
        conv_channels: List[int] = [32, 64, 128],
        cnn_fc_dim: int = 64,
        tabular_in_features: int = 0,
        tab_hidden_dim: int = 32,
        final_fc_dim: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_tabular = tabular_in_features > 0

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves spatial dimensions
        )

        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        # Flattened representation from CNN: conv_channels[2] * 2 * 2
        cnn_flatten_dim = conv_channels[2] * 2 * 2
        self.cnn_dense = nn.Sequential(
            nn.Linear(cnn_flatten_dim, cnn_fc_dim),
            nn.BatchNorm1d(cnn_fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Tabular Context Branch
        if self.use_tabular:
            self.tabular_branch = nn.Sequential(
                nn.Linear(tabular_in_features, tab_hidden_dim),
                nn.BatchNorm1d(tab_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            classifier_in_dim = cnn_fc_dim + tab_hidden_dim
        else:
            self.tabular_branch = None
            classifier_in_dim = cnn_fc_dim

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, final_fc_dim),
            nn.BatchNorm1d(final_fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 1),
        )

    def forward(self, x_matrix: torch.Tensor, x_tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x_matrix: (Batch, 1, H, W)
            x_tabular: (Batch, Tabular_Features) or None

        Returns:
            Logits of shape (Batch,)
        """
        # CNN Branch
        feat = self.conv1(x_matrix)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = torch.flatten(feat, 1)
        cnn_emb = self.cnn_dense(feat)

        # Fusion
        if self.use_tabular and x_tabular is not None:
            tab_emb = self.tabular_branch(x_tabular)
            combined = torch.cat([cnn_emb, tab_emb], dim=1)
        else:
            combined = cnn_emb

        logits = self.classifier(combined).squeeze(-1)
        return logits


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

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

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
    print(f"  Test Samples       : {metrics['test_samples']:,} (Fraud: {metrics['test_fraud_samples']:,})")
    print(f"  Training Time      : {metrics['train_time_sec']:.2f} s")
    print(f"  Inference Time     : {metrics['eval_time_sec']:.3f} s ({metrics['latency_per_sample_ms']:.4f} ms/sample)")
    print("-" * 70)
    print(f"  ROC-AUC Score      : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC Score       : {metrics['pr_auc']:.4f}")
    print(f"  Fraud Precision    : {metrics['precision_fraud']:.4f}")
    print(f"  Fraud Recall       : {metrics['recall_fraud']:.4f}")
    print(f"  Fraud F1-Score     : {metrics['f1_fraud']:.4f}")
    print(f"  Specificity        : {metrics['specificity']:.4f}")
    print(f"  G-Mean Score       : {metrics['g_mean']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Macro F1           : {metrics['f1_macro']:.4f}")
    print("-" * 70)
    print("  Confusion Matrix:")
    print(f"    TN: {cm['tn']:,} | FP: {cm['fp']:,}")
    print(f"    FN: {cm['fn']:,} | TP: {cm['tp']:,}")
    print("=" * 70)


def select_device(requested_device: str) -> torch.device:
    """Select compute device based on availability and user preference."""
    if requested_device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"    Compute device selected: {dev}")
    return dev


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 2D Feature Matrix CNN for E-Commerce Fraud Detection (CNN.pdf)."
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=str,
        default="dataset/e-commerce-fraud-detection/preprocessed-05-09-2026/no_reduction_no_resampling",
        help="Input dataset directory (default: no_reduction_no_resampling).",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Explicit path to train parquet/csv file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Explicit path to test parquet/csv file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/25-09-2026/cnn/no_reduction_no_resampling",
        help="Directory to save artifacts and metrics (default: outputs/25-09-2026/cnn/no_reduction_no_resampling).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="auto",
        help="Target column name ('auto', 'label', 'is_fraud', default: 'auto').",
    )
    parser.add_argument(
        "--user-col",
        type=str,
        default="auto",
        help="User identifier column name ('auto', 'user_id', default: 'auto').",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="auto",
        help="Timestamp column name ('auto', 'timestamp', 'transaction_time', default: 'auto').",
    )
    parser.add_argument(
        "--use-tabular-branch",
        action="store_true",
        default=True,
        help="Enable dual-branch architecture combining 2D CNN with instant tabular features (default: True).",
    )
    parser.add_argument(
        "--no-tabular-branch",
        dest="use_tabular_branch",
        action="store_false",
        help="Disable tabular branch and train purely on 2D Feature Matrix.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (default: 12).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size (default: 512).",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay for AdamW optimizer (default: 0.0001).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout probability (default: 0.25).",
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
        help="Compute device: 'auto', 'cuda', or 'cpu' (default: 'auto').",
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
        help="Disable saving model weights and scalers.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save test probabilities and predictions to test_predictions.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Reproducibility
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
    print(" FRAUD DETECTION MODEL TRAINING: 2D FEATURE MATRIX CNN (CNN.pdf)")
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

    # Detect Columns
    all_cols = train_df.columns
    target_col = detect_column(all_cols, ["label", "is_fraud", "Class", "target"], args.target_col)
    if not target_col:
        raise ValueError(f"Could not find target column in {all_cols}")

    user_col = detect_column(all_cols, ["user_id", "userId", "customer_id", "account_id"], args.user_col)
    time_col = detect_column(all_cols, ["timestamp", "transaction_time", "event_time", "time"], args.time_col)
    amount_col = detect_column(all_cols, ["amount", "transaction_amount", "amt", "price"], "auto") or "amount"
    merch_col = detect_column(all_cols, ["merchant_category", "merchant", "category"], "auto")
    country_col = detect_column(all_cols, ["country", "bin_country"], "auto")
    channel_col = detect_column(all_cols, ["channel", "device", "payment_channel"], "auto")

    # If processed files were passed, try finding corresponding enriched files for spatio-temporal matrices
    if user_col is None or time_col is None:
        print("    Looking for enriched dataset files with user_id and timestamp ...")
        en_candidates = [
            input_path / "artifacts" / "feature_engineered" / "train_enriched.parquet",
            input_path / "feature_engineered" / "train_enriched.parquet",
            input_path.parent / "feature_engineered" / "train_enriched.parquet",
            input_path / "train_enriched.parquet",
        ]
        for c_tr in en_candidates:
            c_te = c_tr.parent / "test_enriched.parquet"
            if c_tr.exists() and c_te.exists():
                print(f"    Found enriched dataset pair: {c_tr.name}")
                train_df = _read_df(c_tr)
                test_df = _read_df(c_te)
                all_cols = train_df.columns
                target_col = detect_column(all_cols, ["label", "is_fraud", "Class"], args.target_col) or target_col
                user_col = detect_column(all_cols, ["user_id", "userId"], "auto")
                time_col = detect_column(all_cols, ["timestamp", "transaction_time"], "auto")
                merch_col = detect_column(all_cols, ["merchant_category", "merchant"], "auto")
                country_col = detect_column(all_cols, ["country", "bin_country"], "auto")
                channel_col = detect_column(all_cols, ["channel"], "auto")
                break

    if user_col is None or time_col is None:
        raise ValueError("CNN 2D Feature Matrix requires 'user_id' and 'timestamp' columns in dataset.")

    y_train = train_df[target_col].to_numpy().astype(int)
    y_test = test_df[target_col].to_numpy().astype(int)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    imbalance_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"    Target Column   : '{target_col}'")
    print(f"    User Column     : '{user_col}'")
    print(f"    Timestamp Column: '{time_col}'")
    print(f"    Train shape     : {train_df.shape} (Neg: {n_neg:,}, Pos: {n_pos:,}, Imbalance ratio: {imbalance_ratio:.2f}:1)")
    print(f"    Test shape      : {test_df.shape} (Fraud: {int(np.sum(y_test == 1)):,})")

    # 1. Build 2D Feature Matrices (H x W)
    X_train_matrix, X_test_matrix, matrix_feat_names = construct_2d_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        user_col=user_col,
        time_col=time_col,
        amount_col=amount_col,
        merch_col=merch_col,
        country_col=country_col,
        channel_col=channel_col,
        windows=DEFAULT_WINDOWS,
    )

    # 2. Build Instantaneous Tabular Context (Wide branch)
    tabular_preprocessor = None
    X_train_tab = None
    X_test_tab = None
    tab_in_dim = 0

    if args.use_tabular_branch:
        ignore_cols = {target_col, "event_id", "transaction_id", user_col, time_col}
        candidate_tab_cols = [c for c in train_df.columns if c not in ignore_cols and not any(w in c for w in DEFAULT_WINDOWS)]
        cat_tab_cols = [c for c in candidate_tab_cols if train_df[c].dtype in (pl.Utf8, pl.Categorical, pl.String)]
        num_tab_cols = [c for c in candidate_tab_cols if c not in cat_tab_cols]

        print(f"    Tabular Context Branch enabled: {len(num_tab_cols)} numeric, {len(cat_tab_cols)} categorical attributes.")
        tabular_preprocessor = TabularContextPreprocessor(num_cols=num_tab_cols, cat_cols=cat_tab_cols)
        X_train_tab = tabular_preprocessor.fit_transform(train_df)
        X_test_tab = tabular_preprocessor.transform(test_df)
        tab_in_dim = X_train_tab.shape[1]

    # Pos weight for imbalanced BCE loss
    if str(args.pos_weight).lower() == "auto":
        pos_weight_val = imbalance_ratio if imbalance_ratio > 2.0 else 1.0
    else:
        pos_weight_val = float(args.pos_weight)

    pos_weight_t = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    # Prepare PyTorch Tensors
    X_tr_mat_t = torch.tensor(X_train_matrix, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)

    X_te_mat_t = torch.tensor(X_test_matrix, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32)

    if args.use_tabular_branch and X_train_tab is not None and X_test_tab is not None:
        X_tr_tab_t = torch.tensor(X_train_tab, dtype=torch.float32)
        X_te_tab_t = torch.tensor(X_test_tab, dtype=torch.float32)
        train_dataset = TensorDataset(X_tr_mat_t, X_tr_tab_t, y_tr_t)
        test_dataset = TensorDataset(X_te_mat_t, X_te_tab_t, y_te_t)
    else:
        train_dataset = TensorDataset(X_tr_mat_t, y_tr_t)
        test_dataset = TensorDataset(X_te_mat_t, y_te_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    # Build FraudCNN2D Model
    grid_height = len(FEATURE_TYPES)
    grid_width = len(DEFAULT_WINDOWS)

    model = FraudCNN2D(
        in_channels=1,
        grid_height=grid_height,
        grid_width=grid_width,
        conv_channels=[32, 64, 128],
        cnn_fc_dim=64,
        tabular_in_features=tab_in_dim,
        tab_hidden_dim=32,
        final_fc_dim=32,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )

    print(
        f"    Model architecture: FraudCNN2D(Grid={grid_height}x{grid_width}, "
        f"Convs=[32, 64, 128], Tabular_In={tab_in_dim}, Pos_Weight={pos_weight_val:.2f})"
    )
    print(f"    Training for {args.epochs} epochs with batch_size={args.batch_size} ...")

    # Training Loop
    t_start_train = time.perf_counter()
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if args.use_tabular_branch:
                batch_mat, batch_tab, batch_y = batch
                batch_mat = batch_mat.to(device)
                batch_tab = batch_tab.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_mat, batch_tab)
            else:
                batch_mat, batch_y = batch
                batch_mat = batch_mat.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_mat)

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_mat.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        scheduler.step()
        avg_loss = total_loss / total_samples

        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(f"      Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.5f} - LR: {current_lr:.6f}")

    train_time = time.perf_counter() - t_start_train

    # Evaluation on Test Set
    print("    Evaluating on test set ...")
    model.eval()
    t_start_eval = time.perf_counter()
    probs_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            if args.use_tabular_branch:
                batch_mat, batch_tab, _ = batch
                batch_mat = batch_mat.to(device)
                batch_tab = batch_tab.to(device)
                logits = model(batch_mat, batch_tab)
            else:
                batch_mat, _ = batch
                batch_mat = batch_mat.to(device)
                logits = model(batch_mat)

            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())

    eval_time = time.perf_counter() - t_start_eval
    y_proba = np.concatenate(probs_list, axis=0)
    y_pred = (y_proba >= args.threshold).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, train_time, eval_time)
    print_evaluation_report(metrics, title=f"EXPERIMENT: {output_dir.name} (FraudCNN2D)")

    # Save metrics JSON directly to output_dir
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "FraudCNN2D (PyTorch)",
                "reference": "CNN.pdf / Yann LeCun 1998 / Zheng et al. CCFD",
                "train_file": str(resolved_train),
                "test_file": str(resolved_test),
                "target_col": target_col,
                "feature_matrix": {
                    "grid_shape": [grid_height, grid_width],
                    "feature_types": FEATURE_TYPES,
                    "windows": DEFAULT_WINDOWS,
                },
                "hyperparameters": {
                    "conv_channels": [32, 64, 128],
                    "cnn_fc_dim": 64,
                    "tabular_in_features": tab_in_dim,
                    "use_tabular_branch": args.use_tabular_branch,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "pos_weight": pos_weight_val,
                    "threshold": args.threshold,
                    "device": str(device),
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Save model weights & preprocessor
    if not args.no_save_model:
        model_file = output_dir / "cnn_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "grid_height": grid_height,
                "grid_width": grid_width,
                "feature_types": FEATURE_TYPES,
                "windows": DEFAULT_WINDOWS,
                "conv_channels": [32, 64, 128],
                "cnn_fc_dim": 64,
                "tabular_in_features": tab_in_dim,
                "use_tabular_branch": args.use_tabular_branch,
                "dropout": args.dropout,
            },
            model_file,
        )

        if tabular_preprocessor is not None:
            preprocessor_file = output_dir / "preprocessor.joblib"
            joblib.dump(tabular_preprocessor, preprocessor_file)
            print(f"    Saved preprocessor to: {preprocessor_file}")

        print(f"    Saved model weights to: {model_file}")

    # Optionally save predictions
    if args.save_predictions:
        pred_file = output_dir / "test_predictions.parquet"
        pred_df = pl.DataFrame({
            "y_true": y_test,
            "y_proba": y_proba,
            "y_pred": y_pred,
        })
        pred_df.write_parquet(pred_file)
        print(f"    Saved test predictions to: {pred_file}")

    print(f"    Saved evaluation metrics to: {metrics_file}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
