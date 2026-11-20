"""Bài 1: tự lan truyền tiến từ trọng số của MLP đã học trên XOR."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("data/xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mlp = MLPClassifier(
    hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0
).fit(X_tr, y_tr)

W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
nam_diem = X_te.iloc[:5]
X5 = nam_diem.to_numpy()

# Lớp ẩn mặc định dùng ReLU; đầu ra phân loại hai lớp dùng sigmoid.
z1 = X5 @ W1 + b1
h = np.maximum(0, z1)
z2 = h @ W2 + b2
xac_suat_tu_tinh = (1 / (1 + np.exp(-z2))).ravel()
xac_suat_sklearn = mlp.predict_proba(nam_diem)[:, 1]

print("Kich thuoc W1, b1, W2, b2:", W1.shape, b1.shape, W2.shape, b2.shape)
print("x1     x2     Tu tinh sigmoid(h @ W2 + b2)   sklearn   Sai lech")
for x, tu_tinh, sklearn in zip(X5, xac_suat_tu_tinh, xac_suat_sklearn):
    print(f"{x[0]:5.2f} {x[1]:6.2f} {tu_tinh:29.12f} {sklearn:10.12f} {abs(tu_tinh-sklearn):10.2e}")
print(f"Sai lech lon nhat: {np.max(np.abs(xac_suat_tu_tinh-xac_suat_sklearn)):.3e}")
print("Nhan xet: W1 va b1 tao 8 gia tri an ReLU; W2 va b2 ket hop 8 gia tri")
print("do thanh mot logit, sigmoid doi logit thanh xac suat lop 1.")
