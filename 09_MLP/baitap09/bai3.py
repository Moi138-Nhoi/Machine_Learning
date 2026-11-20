"""Bài 3: so logistic, tanh và ReLU trên hai vầng trăng."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = {}
print("Activation   Accuracy hoc   Accuracy kiem tra")
for ham in ["logistic", "tanh", "relu"]:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1.0,
                      activation=ham, solver="lbfgs", max_iter=20000,
                      random_state=0),
    ).fit(X_tr, y_tr)
    hoc = accuracy_score(y_tr, mang.predict(X_tr))
    kiem_tra = accuracy_score(y_te, mang.predict(X_te))
    ket_qua[ham] = kiem_tra
    print(f"{ham:10s} {hoc:13.4f} {kiem_tra:19.4f}")

cao_nhat = max(ket_qua.values())
tot_nhat = [ham for ham, acc in ket_qua.items() if acc == cao_nhat]
print("Tot nhat tren tap kiem tra:", ", ".join(tot_nhat), f"({cao_nhat:.4f})")
print("Nhan xet: ham kich hoat doi hinh dang ranh gioi va qua trinh toi uu;")
print("ket qua nay chi ap dung cho cung bo du lieu va tham so thu nghiem.")
