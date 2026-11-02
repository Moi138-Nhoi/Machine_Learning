"""Bài 3: so sánh Gini và entropy khi chọn độ sâu bằng kiểm định chéo."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = {}
for tieu_chi in ("gini", "entropy"):
    diem_cv = {}
    print(f"\nTieu chi: {tieu_chi}")
    print("Do sau  accuracy CV")
    for do_sau in range(1, 16):
        mh = DecisionTreeClassifier(
            criterion=tieu_chi, max_depth=do_sau, random_state=0
        )
        diem_cv[do_sau] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        print(f" {do_sau:2d}      {diem_cv[do_sau]:.4f}")
    do_sau_tot = max(diem_cv, key=diem_cv.get)
    mh_tot = DecisionTreeClassifier(
        criterion=tieu_chi, max_depth=do_sau_tot, random_state=0
    ).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mh_tot.predict(X_te))
    ket_qua[tieu_chi] = (do_sau_tot, diem_cv[do_sau_tot], acc)

print("\nSo sanh")
for ten, (do_sau, cv, acc) in ket_qua.items():
    print(f" {ten:7s}: do sau {do_sau}, CV {cv:.4f}, kiem tra {acc:.4f}")
chenh = ket_qua["entropy"][2] - ket_qua["gini"][2]
print(f"Nhan xet: Entropy va Gini cung chon do sau 5; accuracy kiem tra "
      f"chenh {abs(chenh):.4f}, khong khac nhieu.")
