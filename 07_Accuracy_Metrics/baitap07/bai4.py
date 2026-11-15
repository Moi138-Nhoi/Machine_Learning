"""Bài 4: vẽ ROC của logistic và rừng ngẫu nhiên."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Rung 200 cay": RandomForestClassifier(n_estimators=200, random_state=0),
}
duong = {}
plt.figure(figsize=(7, 6))
for ten, mh in mo_hinh.items():
    xs = mh.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, xs)
    auc = roc_auc_score(y_te, xs)
    duong[ten] = (fpr, tpr, auc)
    plt.plot(fpr, tpr, label=f"{ten}: AUC = {auc:.4f}")
    print(f"{ten}: AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Doan ngau nhien")
plt.xlabel("Ty le bao dong nham (FPR)")
plt.ylabel("Ty le tim dung (TPR/recall)")
plt.title("ROC tren tap kiem tra")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
duong_dan = Path(__file__).resolve().parent / "roc.png"
plt.savefig(duong_dan, dpi=180)
plt.close()
print("Da luu roc.png trong thu muc bai tap.")

# So đường ROC trên hợp các mức FPR, lấy TPR cao nhất ở cùng FPR.
luoi_fpr = np.union1d(duong["Logistic"][0], duong["Rung 200 cay"][0])


def tpr_tai_muc(fpr, tpr, muc):
    vi_tri = np.searchsorted(fpr, muc, side="right") - 1
    return tpr[np.maximum(vi_tri, 0)]


chenh_lech = (
    tpr_tai_muc(*duong["Logistic"][:2], luoi_fpr)
    - tpr_tai_muc(*duong["Rung 200 cay"][:2], luoi_fpr)
)
co_cat_nhau = np.any(chenh_lech > 1e-12) and np.any(chenh_lech < -1e-12)
ten_tot_hon = max(duong, key=lambda ten: duong[ten][2])
print(f"Duong om goc tren trai hon theo AUC: {ten_tot_hon}")
print("Hai duong co doan vuot qua nhau:", bool(co_cat_nhau))
print(
    "Nhan xet: AUC cho biet kha nang xep hang tren moi nguong; "
    "duong ROC co the doi thu tu tai tung muc bao dong nham."
)
