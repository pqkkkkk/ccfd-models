"""Graph Construction Script for Fraud Detection.

Implements two graph construction strategies:
- Strategy 1 (Homogeneous Temporal Graph): Intra-user transaction sequence graph with directed edges
  connecting past transactions to subsequent transactions (T_{t-k} -> T_t).
- Strategy 3 (Heterogeneous Bipartite Graph): Bipartite graph between User nodes and Transaction nodes
  (user <-> transaction).

Prevents data leakage by:
1. Sorting by transaction_time and splitting chronologically (Train / Val / Test).
2. Fitting standard scalers and encoders ONLY on the training split.
3. Ensuring edges in the temporal graph flow forward in time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data, HeteroData


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct PyG Graphs from Fraud Detection Raw Data")
    parser.add_argument(
        "--raw-path",
        type=str,
        default="dataset/e-commerce-fraud-detection/raw.csv",
        help="Path to raw.csv dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/e-commerce-fraud-detection/graphs",
        help="Directory to save generated graph files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of data for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Proportion of data for validation (default: 0.10)",
    )
    parser.add_argument(
        "--max-prev-k",
        type=int,
        default=2,
        help="Number of preceding user transactions to connect in Strategy 1 (default: 2)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["temporal", "bipartite"],
        choices=["temporal", "bipartite"],
        help="Which graph strategies to build ('temporal', 'bipartite', or both)",
    )
    return parser.parse_args()


def load_and_preprocess(
    raw_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load raw dataset, sort by time, compute features, and split masks."""
    print(f">>> Reading raw dataset from: {raw_path}")
    df = pl.read_csv(raw_path)

    # 1. Parse datetime & sort chronologically
    df = df.with_columns(
        pl.col("transaction_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ").alias("dt")
    )
    df = df.sort("dt")

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_mask = np.zeros(total_rows, dtype=bool)
    val_mask = np.zeros(total_rows, dtype=bool)
    test_mask = np.zeros(total_rows, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    # 2. Feature engineering
    df = df.with_columns([
        (pl.col("country") != pl.col("bin_country")).cast(pl.Float32).alias("country_mismatch"),
        (pl.col("amount") / (pl.col("avg_amount_user") + 1e-4)).alias("amount_to_avg_ratio"),
        (pl.col("amount") - pl.col("avg_amount_user")).alias("amount_diff_avg"),
        (pl.col("channel") == "web").cast(pl.Float32).alias("is_channel_web"),
        pl.col("dt").dt.hour().cast(pl.Float32).alias("hour_of_day"),
        pl.col("dt").dt.weekday().cast(pl.Float32).alias("day_of_week"),
    ])

    # Convert binary security flags to float
    for col in ["promo_used", "avs_match", "cvv_result", "three_ds_flag"]:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

    num_cols = [
        "amount",
        "shipping_distance_km",
        "account_age_days",
        "total_transactions_user",
        "avg_amount_user",
        "amount_to_avg_ratio",
        "amount_diff_avg",
        "hour_of_day",
        "day_of_week",
    ]

    bin_cols = [
        "country_mismatch",
        "is_channel_web",
        "promo_used",
        "avs_match",
        "cvv_result",
        "three_ds_flag",
    ]

    cat_cols = ["country", "bin_country", "merchant_category"]

    # Scale numeric columns (fit ONLY on train)
    scaler = StandardScaler()
    num_train = df.select(num_cols).filter(pl.Series(train_mask)).to_numpy()
    scaler.fit(num_train)
    num_all = scaler.transform(df.select(num_cols).to_numpy())

    # One-hot encode categorical columns (fit on full domain or train)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_all_raw = df.select(cat_cols).to_numpy()
    cat_all = ohe.fit_transform(cat_all_raw)

    bin_all = df.select(bin_cols).to_numpy()

    # Combine all node features
    X = np.hstack([num_all, bin_all, cat_all]).astype(np.float32)
    y = df["is_fraud"].to_numpy().astype(np.int64)

    feature_names = (
        num_cols
        + bin_cols
        + list(ohe.get_feature_names_out(cat_cols))
    )

    print(f"    Total samples : {total_rows:,}")
    print(f"    Train samples : {train_mask.sum():,} ({df.filter(pl.Series(train_mask))['is_fraud'].mean()*100:.2f}% fraud)")
    print(f"    Val samples   : {val_mask.sum():,} ({df.filter(pl.Series(val_mask))['is_fraud'].mean()*100:.2f}% fraud)")
    print(f"    Test samples  : {test_mask.sum():,} ({df.filter(pl.Series(test_mask))['is_fraud'].mean()*100:.2f}% fraud)")
    print(f"    Node feature dimension: {X.shape[1]}")

    return df, X, y, train_mask, val_mask, test_mask, feature_names


def build_temporal_graph(
    df: pl.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    max_prev_k: int = 2,
) -> Data:
    """Build Strategy 1: Homogeneous Temporal Graph connecting past user transactions to future ones."""
    print("\n>>> Building Strategy 1: Homogeneous Temporal Graph ...")
    t0 = time.perf_counter()

    # Create mapping from df row index to node index
    # Note: df is already sorted by time
    df_with_idx = df.with_columns(pl.int_range(0, len(df)).alias("node_idx"))

    # Group by user_id to extract chronological transaction sequences
    src_nodes: List[int] = []
    dst_nodes: List[int] = []
    time_deltas: List[float] = []
    amount_deltas: List[float] = []

    user_groups = df_with_idx.group_by("user_id", maintain_order=True).agg([
        pl.col("node_idx"),
        pl.col("dt"),
        pl.col("amount"),
    ])

    for row in user_groups.iter_rows():
        indices = row[1]
        datetimes = row[2]
        amounts = row[3]
        n_tx = len(indices)

        for i in range(1, n_tx):
            # Connect up to max_prev_k previous transactions
            for k in range(1, min(i + 1, max_prev_k + 1)):
                prev_i = i - k
                src_nodes.append(indices[prev_i])
                dst_nodes.append(indices[i])

                dt_sec = max(0.0, (datetimes[i] - datetimes[prev_i]).total_seconds())
                amt_diff = abs(amounts[i] - amounts[prev_i])

                time_deltas.append(np.log1p(dt_sec))
                amount_deltas.append(np.log1p(amt_diff))

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(
        np.column_stack([time_deltas, amount_deltas]), dtype=torch.float
    )

    data = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )

    elapsed = time.perf_counter() - t0
    print(f"    Nodes       : {data.num_nodes:,}")
    print(f"    Edges       : {data.num_edges:,}")
    print(f"    Edge features: {edge_attr.shape[1]} (log_dt, log_damount)")
    print(f"    Completed in: {elapsed:.2f}s")
    return data


def build_bipartite_graph(
    df: pl.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> HeteroData:
    """Build Strategy 3: Heterogeneous Bipartite Graph between User and Transaction nodes."""
    print("\n>>> Building Strategy 3: Heterogeneous Bipartite Graph ...")
    t0 = time.perf_counter()

    # Map user_id to 0..num_users-1
    unique_users = df["user_id"].unique().sort().to_list()
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    num_users = len(unique_users)

    # Compute User static/profile features
    user_df = df.group_by("user_id").agg([
        pl.col("account_age_days").first().alias("account_age_days"),
        pl.col("total_transactions_user").first().alias("total_transactions_user"),
        pl.col("avg_amount_user").first().alias("avg_amount_user"),
    ]).sort("user_id")

    user_scaler = StandardScaler()
    user_feats = user_scaler.fit_transform(
        user_df.select(["account_age_days", "total_transactions_user", "avg_amount_user"]).to_numpy()
    ).astype(np.float32)

    # Transaction nodes: 0..N-1
    tx_indices = np.arange(len(df), dtype=np.int64)
    tx_users = np.array([user_to_idx[uid] for uid in df["user_id"].to_list()], dtype=np.int64)

    # Edges: user -> transaction & transaction -> user
    u_to_tx_src = torch.from_numpy(tx_users)
    u_to_tx_dst = torch.from_numpy(tx_indices)

    hetero_data = HeteroData()

    # Add node features
    hetero_data["user"].x = torch.from_numpy(user_feats)
    hetero_data["user"].num_nodes = num_users

    hetero_data["transaction"].x = torch.from_numpy(X)
    hetero_data["transaction"].y = torch.from_numpy(y)
    hetero_data["transaction"].train_mask = torch.from_numpy(train_mask)
    hetero_data["transaction"].val_mask = torch.from_numpy(val_mask)
    hetero_data["transaction"].test_mask = torch.from_numpy(test_mask)

    # Add edges
    hetero_data["user", "performs", "transaction"].edge_index = torch.stack([u_to_tx_src, u_to_tx_dst], dim=0)
    hetero_data["transaction", "performed_by", "user"].edge_index = torch.stack([u_to_tx_dst, u_to_tx_src], dim=0)

    elapsed = time.perf_counter() - t0
    print(f"    User nodes        : {num_users:,}")
    print(f"    Transaction nodes : {len(df):,}")
    print(f"    Edges (performs)  : {hetero_data['user', 'performs', 'transaction'].num_edges:,}")
    print(f"    Completed in      : {elapsed:.2f}s")
    return hetero_data


def main() -> None:
    args = parse_arguments()
    raw_path = Path(args.raw_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    print("=" * 70)
    print(" GRAPH CONSTRUCTION FOR FRAUD DETECTION")
    print(f" Raw dataset : {raw_path}")
    print(f" Output dir  : {output_dir}")
    print(f" Train/Val/Test: {args.train_ratio:.2f} / {args.val_ratio:.2f} / {1.0 - args.train_ratio - args.val_ratio:.2f}")
    print("=" * 70)

    df, X, y, train_mask, val_mask, test_mask, feature_names = load_and_preprocess(
        raw_path=raw_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Save feature names metadata
    metadata = {
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    torch.save(metadata, output_dir / "features_metadata.pt")

    if "temporal" in args.strategies:
        temporal_graph = build_temporal_graph(
            df=df,
            X=X,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            max_prev_k=args.max_prev_k,
        )
        temp_path = output_dir / "temporal_graph.pt"
        torch.save(temporal_graph, temp_path)
        print(f"    Saved temporal graph to: {temp_path}")

    if "bipartite" in args.strategies:
        bipartite_graph = build_bipartite_graph(
            df=df,
            X=X,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        bipartite_path = output_dir / "bipartite_graph.pt"
        torch.save(bipartite_graph, bipartite_path)
        print(f"    Saved bipartite graph to: {bipartite_path}")

    print("\nGraph construction completed successfully!")


if __name__ == "__main__":
    main()
