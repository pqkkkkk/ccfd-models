"""
Model Performance Comparison Visualization
Compares models across evaluation dates:
- 09-05-2026: FFNN, Random Forest, XGBoost (no_reduction_no_resampling)
- 09-25-2026: CNN, LSTM (no_reduction_no_resampling)
- 10-04-2026: Temporal GAT, Temporal GraphSAGE
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {
        "id": "rf",
        "name": "Random Forest",
        "category": "Traditional ML",
        "color": "#2ca02c",
        "path": BASE_DIR / "outputs" / "09-05-2026" / "random_forest" / "no_reduction_no_resampling" / "metrics.json",
    },
    {
        "id": "xgb",
        "name": "XGBoost",
        "category": "Traditional ML",
        "color": "#1f77b4",
        "path": BASE_DIR / "outputs" / "09-05-2026" / "xgboost" / "no_reduction_no_resampling" / "metrics.json",
    },
    {
        "id": "ffnn",
        "name": "FFNN",
        "category": "Deep Learning (Tabular)",
        "color": "#ff7f0e",
        "path": BASE_DIR / "outputs" / "09-05-2026" / "feed_forward_nn" / "no_reduction_no_resampling" / "metrics.json",
    },
    {
        "id": "cnn",
        "name": "CNN (FraudCNN2D)",
        "category": "Deep Learning (Grid)",
        "color": "#9467bd",
        "path": BASE_DIR / "outputs" / "09-25-2026" / "cnn" / "no_reduction_no_resampling" / "metrics.json",
    },
    {
        "id": "lstm",
        "name": "LSTM (Sequence)",
        "category": "Deep Learning (Sequential)",
        "color": "#8c564b",
        "path": BASE_DIR / "outputs" / "09-25-2026" / "lstm" / "no_reduction_no_resampling" / "metrics.json",
    },
    {
        "id": "gat",
        "name": "Temporal GAT",
        "category": "Graph Neural Network",
        "color": "#e377c2",
        "path": BASE_DIR / "outputs" / "10-04-2026" / "gnn" / "temporal" / "gat" / "metrics.json",
    },
    {
        "id": "graphsage",
        "name": "Temporal GraphSAGE",
        "category": "Graph Neural Network",
        "color": "#17becf",
        "path": BASE_DIR / "outputs" / "10-04-2026" / "gnn" / "temporal" / "graphsage" / "metrics.json",
    },
]

def load_metrics():
    models_data = []
    for cfg in MODEL_CONFIGS:
        p = cfg["path"]
        if not p.exists():
            print(f"Warning: File not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data["metrics"]
        models_data.append({
            **cfg,
            "roc_auc": m.get("roc_auc", 0.0),
            "pr_auc": m.get("pr_auc", 0.0),
            "f1_fraud": m.get("f1_fraud", 0.0),
            "precision_fraud": m.get("precision_fraud", 0.0),
            "recall_fraud": m.get("recall_fraud", 0.0),
            "accuracy": m.get("accuracy", 0.0),
            "specificity": m.get("specificity", 0.0),
            "g_mean": m.get("g_mean", 0.0),
            "f1_macro": m.get("f1_macro", 0.0),
            "f1_weighted": m.get("f1_weighted", 0.0),
            "train_time_sec": m.get("train_time_sec", 0.0),
            "eval_time_sec": m.get("eval_time_sec", 0.0),
            "latency_ms": m.get("latency_per_sample_ms", 0.0),
            "tp": m.get("confusion_matrix", {}).get("tp", 0),
            "fp": m.get("confusion_matrix", {}).get("fp", 0),
            "fn": m.get("confusion_matrix", {}).get("fn", 0),
            "tn": m.get("confusion_matrix", {}).get("tn", 0),
            "test_samples": m.get("test_samples", 0),
            "test_fraud_samples": m.get("test_fraud_samples", 0),
        })
    return models_data

def plot_primary_metrics_grouped(models_data, save_path):
    """Bar chart comparing major metrics across all models."""
    metrics_to_plot = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("F1 (Fraud)", "f1_fraud"),
        ("Precision", "precision_fraud"),
        ("Recall", "recall_fraud"),
        ("G-Mean", "g_mean"),
    ]
    
    n_models = len(models_data)
    n_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    for i, m in enumerate(models_data):
        offset = (i - n_models / 2 + 0.5) * width
        vals = [m[key] for _, key in metrics_to_plot]
        bars = ax.bar(x + offset, vals, width, label=m["name"], color=m["color"], edgecolor="black", linewidth=0.6, alpha=0.9)
        # Value labels above bars
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}",
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight='bold')
    ax.set_title("Performance Metrics Comparison Across Fraud Detection Models", fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics_to_plot], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_radar_chart(models_data, save_path):
    """Radar chart highlighting the balance among key metrics."""
    categories = ['ROC-AUC', 'PR-AUC', 'F1-Fraud', 'Precision', 'Recall', 'Specificity', 'G-Mean']
    keys = ['roc_auc', 'pr_auc', 'f1_fraud', 'precision_fraud', 'recall_fraud', 'specificity', 'g_mean']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True), dpi=300)
    
    plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    for m in models_data:
        values = [m[k] for k in keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=m['name'], color=m['color'])
        ax.fill(angles, values, color=m['color'], alpha=0.08)
        
    plt.title("Multi-Metric Radar Profile Comparison", size=15, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_precision_recall_tradeoff(models_data, save_path):
    """Scatter plot of Precision vs Recall with bubble size proportional to F1-Score."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Iso-F1 curves background
    f_scores = np.linspace(0.2, 0.8, 4)
    for f_score in f_scores:
        x_curve = np.linspace(f_score / (2 - f_score) + 0.001, 1, 100)
        y_curve = f_score * x_curve / (2 * x_curve - f_score)
        valid = (y_curve >= 0) & (y_curve <= 1)
        ax.plot(x_curve[valid], y_curve[valid], color='gray', alpha=0.3, linestyle=':')
        ax.annotate(f'F1={f_score:.1f}', xy=(1.01, f_score / (2 - f_score)), color='gray', fontsize=8, alpha=0.6)
    
    for m in models_data:
        rec = m["recall_fraud"]
        prec = m["precision_fraud"]
        f1 = m["f1_fraud"]
        size = (f1 ** 2) * 1200 + 100
        
        ax.scatter(rec, prec, s=size, color=m["color"], alpha=0.75, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(
            f"{m['name']}\n(P={prec:.2f}, R={rec:.2f}, F1={f1:.2f})",
            xy=(rec, prec),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=m["color"], lw=1.2, alpha=0.85),
            zorder=6
        )
        
    ax.set_xlabel("Recall (Fraud Detection Rate)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision (Fraud Positive Predictive Value)", fontsize=12, fontweight='bold')
    ax.set_title("Fraud Detection Trade-Off: Precision vs Recall\n(Bubble size indicates F1-Score on Fraud Class)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.1, 0.6)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_computational_efficiency(models_data, save_path):
    """Bar charts for Training Time and Inference Latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    names = [m["name"] for m in models_data]
    colors = [m["color"] for m in models_data]
    train_times = [m["train_time_sec"] for m in models_data]
    latencies = [m["latency_ms"] for m in models_data]
    
    # 1. Training Time (Log scale due to large difference between XGB 1.8s and CNN 427s)
    y_pos = np.arange(len(names))
    bars1 = ax1.barh(y_pos, train_times, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xscale('log')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10, fontweight='bold')
    ax1.set_xlabel("Training Time (seconds, Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_title("Model Training Time", fontsize=13, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w * 1.15, bar.get_y() + bar.get_height() / 2, f"{w:.1f}s",
                 va='center', fontsize=9, fontweight='bold')
        
    # 2. Inference Latency per sample (ms)
    bars2 = ax2.barh(y_pos, latencies, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Inference Latency (ms / sample)", fontsize=11, fontweight='bold')
    ax2.set_title("Inference Latency per Sample", fontsize=13, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f} ms",
                 va='center', fontsize=9, fontweight='bold')
        
    plt.suptitle("Computational Efficiency & Scalability Comparison", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_breakdown(models_data, save_path):
    """Normalized Confusion Matrix breakdown (Detection Rate TP% vs False Alarm Rate FP%)."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    names = [m["name"] for m in models_data]
    # TP rate = recall
    tp_rates = [m["recall_fraud"] * 100 for m in models_data]
    # FP rate = 1 - specificity
    fp_rates = [(1.0 - m["specificity"]) * 100 for m in models_data]
    
    x = np.arange(len(names))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, tp_rates, width, label='True Positive Rate / Recall % (Fraud Detected ↑)',
                    color='#2ca02c', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, fp_rates, width, label='False Positive Rate % (False Alarm on Legitimate ↓)',
                    color='#d62728', edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Operational Impact: Detection Rate vs False Alarm Rate', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9.5, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def generate_markdown_report(models_data, report_path):
    lines = []
    lines.append("# Báo Cáo So Sánh Hiệu Suất Mô Hình Phát Hiện Gian Lận (CCFD)\n")
    lines.append("## 1. Bảng Tổng Hợp Metrics Chi Tiết\n")
    lines.append("| Mô hình | Nhóm kiến trúc | ROC-AUC | PR-AUC | F1 (Fraud) | Precision | Recall | Specificity | G-Mean | Accuracy | Train Time | Latency/sample |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    metric_keys = ['roc_auc', 'pr_auc', 'f1_fraud', 'precision_fraud', 'recall_fraud', 'specificity', 'g_mean', 'accuracy']
    max_vals = {k: max(m[k] for m in models_data) for k in metric_keys}

    def fmt(val, key):
        s = f"{val:.4f}"
        if abs(val - max_vals[key]) < 1e-5:
            return f"**{s}**"
        return s

    for m in models_data:
        roc_str = fmt(m['roc_auc'], 'roc_auc')
        pr_str = fmt(m['pr_auc'], 'pr_auc')
        f1_str = fmt(m['f1_fraud'], 'f1_fraud')
        prec_str = fmt(m['precision_fraud'], 'precision_fraud')
        rec_str = fmt(m['recall_fraud'], 'recall_fraud')
        spec_str = fmt(m['specificity'], 'specificity')
        gm_str = fmt(m['g_mean'], 'g_mean')
        acc_str = fmt(m['accuracy'], 'accuracy')

        lines.append(
            f"| **{m['name']}** | {m['category']} | {roc_str} | {pr_str} | "
            f"{f1_str} | {prec_str} | {rec_str} | "
            f"{spec_str} | {gm_str} | {acc_str} | "
            f"{m['train_time_sec']:.2f}s | {m['latency_ms']:.4f}ms |"
        )
    
    lines.append("\n## 2. Chi Tiết Confusion Matrix\n")
    lines.append("| Mô hình | Test Samples | Fraud Samples | TP (Bắt trúng) | FP (Báo nhầm) | FN (Bỏ lọt) | TN (Đúng hợp lệ) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models_data:
        lines.append(
            f"| **{m['name']}** | {m['test_samples']} | {m['test_fraud_samples']} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
        
    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved Report: {report_path}")
    return content

def main():
    models_data = load_metrics()
    print(f"Loaded metrics for {len(models_data)} models:")
    for m in models_data:
        print(f" - {m['name']} ({m['category']}): ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, F1-Fraud={m['f1_fraud']:.4f}")
        
    # Generate all plots
    p1 = OUTPUT_DIR / "1_primary_metrics_comparison.png"
    p2 = OUTPUT_DIR / "2_radar_profile_comparison.png"
    p3 = OUTPUT_DIR / "3_precision_recall_tradeoff.png"
    p4 = OUTPUT_DIR / "4_computational_efficiency.png"
    p5 = OUTPUT_DIR / "5_detection_vs_false_alarms.png"
    report_file = OUTPUT_DIR / "comparison_summary.md"
    
    plot_primary_metrics_grouped(models_data, p1)
    plot_radar_chart(models_data, p2)
    plot_precision_recall_tradeoff(models_data, p3)
    plot_computational_efficiency(models_data, p4)
    plot_confusion_breakdown(models_data, p5)
    
    md_content = generate_markdown_report(models_data, report_file)
    print("Report generated successfully.")

if __name__ == "__main__":
    main()
