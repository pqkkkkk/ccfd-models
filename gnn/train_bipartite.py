"""Training & Evaluation Script for Strategy 3 (Heterogeneous Bipartite Graph).

Trains HeteroBipartiteGNN on User <-> Transaction interactions using fast GPU execution.
Supports automated class-weighting, early stopping on Val PR-AUC, and comprehensive metrics export.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData

from gnn.models import HeteroBipartiteGNN
from gnn.utils import evaluate_predictions, print_evaluation_report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN on Heterogeneous Bipartite Graph")
    parser.add_argument(
        "--graph-path",
        type=str,
        default="dataset/e-commerce-fraud-detection/graphs/bipartite_graph.pt",
        help="Path to bipartite_graph.pt",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of HeteroConv layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping based on Val PR-AUC (default: 10)",
    )
    parser.add_argument(
        "--pos-weight",
        type=str,
        default="auto",
        help="Positive class weight: 'auto', 'none', or float value (default: 'auto')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gnn/bipartite",
        help="Output directory for metrics and artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    graph_path = Path(args.graph_path).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}. Please run 'python -m gnn.build_graphs' first."
        )

    # Determine device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir) / "hetero_sage"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" GNN FRAUD DETECTION TRAINING: HETEROGENEOUS BIPARTITE GRAPH")
    print(f" Graph File : {graph_path}")
    print(f" Device     : {device}")
    print(f" Output Dir : {output_dir}")
    print("=" * 70)

    print(">>> Loading HeteroData ...")
    data: HeteroData = torch.load(graph_path, weights_only=False)

    num_users = data["user"].num_nodes
    num_tx = data["transaction"].num_nodes
    user_in_dim = data["user"].x.shape[1]
    tx_in_dim = data["transaction"].x.shape[1]

    train_mask = data["transaction"].train_mask.to(device)
    val_mask = data["transaction"].val_mask.to(device)
    test_mask = data["transaction"].test_mask.to(device)

    y_train = data["transaction"].y[data["transaction"].train_mask].numpy()
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    imbalance_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"    User nodes        : {num_users:,} (dim: {user_in_dim})")
    print(f"    Transaction nodes : {num_tx:,} (dim: {tx_in_dim})")
    print(f"    Train             : {train_mask.sum().item():,} (Fraud: {n_pos:,}, Normal: {n_neg:,}, Ratio: {imbalance_ratio:.2f}:1)")
    print(f"    Val               : {val_mask.sum().item():,}")
    print(f"    Test              : {test_mask.sum().item():,}")

    # Move tensors to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    y = data["transaction"].y.to(device).float()

    # Initialize model
    print(">>> Initializing HeteroBipartiteGNN ...")
    model = HeteroBipartiteGNN(
        user_in_dim=user_in_dim,
        tx_in_dim=tx_in_dim,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Loss function with pos_weight
    if args.pos_weight == "auto":
        weight_val = np.sqrt(imbalance_ratio)
        pos_weight_val = torch.tensor([weight_val], device=device, dtype=torch.float)
        print(f"    Auto pos_weight: {weight_val:.2f} (sqrt of imbalance ratio {imbalance_ratio:.2f})")
    elif args.pos_weight == "none":
        pos_weight_val = None
    else:
        pos_weight_val = torch.tensor([float(args.pos_weight)], device=device, dtype=torch.float)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_prauc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None

    print(f">>> Starting training for {args.epochs} epochs ...")
    t_start_train = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0_ep = time.perf_counter()

        # Training
        model.train()
        optimizer.zero_grad()
        logits = model(x_dict, edge_index_dict)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out_val = model(x_dict, edge_index_dict)
            val_proba = torch.sigmoid(out_val[val_mask]).cpu().numpy()
            y_val_true = y[val_mask].cpu().numpy().astype(int)
            y_val_pred = (val_proba >= 0.5).astype(int)

        val_metrics = evaluate_predictions(y_val_true, y_val_pred, val_proba, 0.0, 0.0)
        val_prauc = val_metrics["pr_auc"]
        val_f1 = val_metrics["f1_fraud"]
        val_roc = val_metrics["roc_auc"]

        scheduler.step(val_prauc)
        ep_duration = time.perf_counter() - t0_ep

        print(
            f" Epoch {epoch:02d}/{args.epochs:02d} [{ep_duration*1000:.1f}ms] | "
            f"Loss: {loss.item():.4f} | "
            f"Val PR-AUC: {val_prauc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val ROC-AUC: {val_roc:.4f}"
        )

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f">>> Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch})")
                break

    train_total_time = time.perf_counter() - t_start_train

    # Load best model for test evaluation
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    print(f"\n>>> Evaluating Best Model (Epoch {best_epoch}) on Test Set ...")
    model.eval()
    t_start_eval = time.perf_counter()
    with torch.no_grad():
        out_test = model(x_dict, edge_index_dict)
        test_proba = torch.sigmoid(out_test[test_mask]).cpu().numpy()
        y_test_true = y[test_mask].cpu().numpy().astype(int)
        y_test_pred = (test_proba >= 0.5).astype(int)
    eval_total_time = time.perf_counter() - t_start_eval

    test_metrics = evaluate_predictions(
        y_true=y_test_true,
        y_pred=y_test_pred,
        y_proba=test_proba,
        train_time=train_total_time,
        eval_time=eval_total_time,
    )

    print_evaluation_report(test_metrics, title="EXPERIMENT: Heterogeneous Bipartite GraphSAGE")

    # Save metrics JSON directly to output_dir
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "HeteroBipartiteGNN",
                "strategy": "Strategy 3: Heterogeneous Bipartite Graph (User <-> Transaction)",
                "graph_file": str(graph_path),
                "hyperparameters": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "best_epoch": best_epoch,
                    "pos_weight": str(args.pos_weight),
                },
                "feature_count_user": user_in_dim,
                "feature_count_tx": tx_in_dim,
                "metrics": test_metrics,
            },
            f,
            indent=2,
        )

    # Save model checkpoint
    model_file = output_dir / "hetero_sage_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_in_dim": user_in_dim,
            "tx_in_dim": tx_in_dim,
            "hidden_dim": args.hidden_dim,
            "best_epoch": best_epoch,
        },
        model_file,
    )
    print(f"    Saved model checkpoint to: {model_file}")
    print(f"    Saved metrics to: {metrics_file}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
