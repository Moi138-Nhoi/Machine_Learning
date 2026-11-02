"""Bài 2: tự tìm câu hỏi gốc bằng Gini trên 450 hồ sơ tập học."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


thu_muc_goc = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(thu_muc_goc))
from f1_gini import gini, gini_sau_chia, tim_cau_hoi


df = pd.read_csv(thu_muc_goc / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, _, y_tr, _ = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ten_cot, nguong = tim_cau_hoi(X_tr, y_tr)
di_trai = X_tr[ten_cot].to_numpy() <= nguong
g_goc = gini(y_tr)
g_chia = gini_sau_chia(y_tr, di_trai)
print(f"So khach tap hoc: {len(X_tr)}")
print(f"Cau hoi tot nhat: {ten_cot} <= {nguong:.2f}")
print(f"So khach nhanh trai/phai: {di_trai.sum()}/{(~di_trai).sum()}")
print(f"Gini goc: {g_goc:.4f}; Gini sau chia: {g_chia:.4f}; "
      f"Gini giam: {g_goc - g_chia:.4f}")

cay = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
cot_sklearn = X_tr.columns[cay.tree_.feature[0]]
nguong_sklearn = cay.tree_.threshold[0]
print(f"scikit-learn: {cot_sklearn} <= {nguong_sklearn:.2f}")
print("Giai thich: Thu trung diem cua moi cap gia tri ke nhau o tung cot; "
      "tinh Gini co trong so cho hai nhanh, roi chon cach chia lam Gini giam nhieu nhat.")
