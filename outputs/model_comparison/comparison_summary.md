# Báo Cáo So Sánh Hiệu Suất Mô Hình Phát Hiện Gian Lận (CCFD)

## 1. Bảng Tổng Hợp Metrics Chi Tiết

| Mô hình | Nhóm kiến trúc | ROC-AUC | PR-AUC | F1 (Fraud) | Precision | Recall | Specificity | G-Mean | Accuracy | Train Time | Latency/sample |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Traditional ML | 0.9753 | 0.7844 | 0.5264 | 0.3806 | 0.8531 | 0.9691 | 0.9093 | 0.9666 | 6.41s | 0.0042ms |
| **XGBoost** | Traditional ML | **0.9761** | **0.8485** | 0.4646 | 0.3143 | **0.8905** | 0.9568 | **0.9231** | 0.9554 | 1.87s | 0.0011ms |
| **FFNN** | Deep Learning (Tabular) | 0.9601 | 0.7193 | 0.3154 | 0.1921 | 0.8808 | 0.9177 | 0.8991 | 0.9169 | 63.82s | 0.0042ms |
| **CNN (FraudCNN2D)** | Deep Learning (Grid) | 0.9627 | 0.7463 | 0.3186 | 0.1941 | 0.8890 | 0.9180 | 0.9034 | 0.9174 | 427.44s | 0.0671ms |
| **LSTM (Sequence)** | Deep Learning (Sequential) | 0.9595 | 0.7188 | 0.3966 | 0.2593 | 0.8429 | 0.9465 | 0.8932 | 0.9443 | 29.85s | 0.0036ms |
| **Temporal GAT** | Graph Neural Network | 0.8608 | 0.2853 | 0.3890 | 0.3037 | 0.5409 | 0.9723 | 0.7252 | 0.9629 | 191.30s | 0.0230ms |
| **Temporal GraphSAGE** | Graph Neural Network | 0.9468 | 0.6476 | **0.5561** | **0.4761** | 0.6684 | **0.9836** | 0.8108 | **0.9767** | 13.55s | 0.0024ms |

## 2. Chi Tiết Confusion Matrix

| Mô hình | Test Samples | Fraud Samples | TP (Bắt trúng) | FP (Báo nhầm) | FN (Bỏ lọt) | TN (Đúng hợp lệ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 89909 | 1954 | 1667 | 2713 | 287 | 85242 |
| **XGBoost** | 89909 | 1954 | 1740 | 3796 | 214 | 84159 |
| **FFNN** | 89909 | 1954 | 1721 | 7237 | 233 | 80718 |
| **CNN (FraudCNN2D)** | 89909 | 1954 | 1737 | 7212 | 217 | 80743 |
| **LSTM (Sequence)** | 89909 | 1954 | 1647 | 4704 | 307 | 83251 |
| **Temporal GAT** | 59940 | 1309 | 708 | 1623 | 601 | 57008 |
| **Temporal GraphSAGE** | 59940 | 1309 | 875 | 963 | 434 | 57668 |