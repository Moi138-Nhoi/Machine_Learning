"""Bài 1: tự tính bốn thước đo phân loại bằng NumPy."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Trả về accuracy, precision, recall, F1 của lớp 1; mẫu số 0 nhận 0."""
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)
    if that.shape != doan.shape or that.size == 0:
        raise ValueError("Hai mang nhan phai cung kich thuoc va khong rong")

    tp = np.sum((that == 1) & (doan == 1))
    fp = np.sum((that == 0) & (doan == 1))
    fn = np.sum((that == 1) & (doan == 0))
    tn = np.sum((that == 0) & (doan == 0))
    accuracy = (tp + tn) / that.size
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Hoi quy logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
}
for ten, mh in mo_hinh.items():
    du_doan = mh.fit(X_tr, y_tr).predict(X_te)
    tp = np.sum((y_te.to_numpy() == 1) & (du_doan == 1))
    fp = np.sum((y_te.to_numpy() == 0) & (du_doan == 1))
    fn = np.sum((y_te.to_numpy() == 1) & (du_doan == 0))
    tn = np.sum((y_te.to_numpy() == 0) & (du_doan == 0))
    tu_tinh = danh_gia(y_te, du_doan)
    thu_vien = (
        accuracy_score(y_te, du_doan),
        precision_score(y_te, du_doan, zero_division=0),
        recall_score(y_te, du_doan, zero_division=0),
        f1_score(y_te, du_doan, zero_division=0),
    )
    print(f"{ten}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("  Tu tinh  : accuracy={:.4f}, precision={:.4f}, recall={:.4f}, F1={:.4f}".format(*tu_tinh))
    print("  sklearn  : accuracy={:.4f}, precision={:.4f}, recall={:.4f}, F1={:.4f}".format(*thu_vien))
    print("  Hai cach trung nhau:", np.allclose(tu_tinh, thu_vien))

print(
    "Nhan xet: mo hinh luoi khong du doan lop 1, nen TP + FP = 0. "
    "Precision = TP/(TP+FP) la 0/0, khong xac dinh ve toan hoc. "
    "Ham quy uoc tra 0 de khong loi chia cho 0, giong zero_division=0 cua sklearn."
)
