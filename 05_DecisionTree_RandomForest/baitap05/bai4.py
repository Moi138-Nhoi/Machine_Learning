"""Bài 4: chọn min_samples_leaf bằng CV và so với cây sâu 5."""

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

diem = {}
print("min_samples_leaf  accuracy CV")
for so_khach in (1, 2, 5, 10, 20, 40):
    mh = DecisionTreeClassifier(min_samples_leaf=so_khach, random_state=0)
    diem[so_khach] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    print(f" {so_khach:2d}               {diem[so_khach]:.4f}")

tot = max(diem, key=diem.get)
cay_la = DecisionTreeClassifier(min_samples_leaf=tot, random_state=0).fit(X_tr, y_tr)
cay_sau = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr, y_tr)
acc_la = accuracy_score(y_te, cay_la.predict(X_te))
acc_sau = accuracy_score(y_te, cay_sau.predict(X_te))
print(f"\nTot nhat: min_samples_leaf={tot}, CV={diem[tot]:.4f}")
print(f"Accuracy tap kiem tra, cat tia theo la: {acc_la:.4f}")
print(f"Accuracy tap kiem tra, cay sau 5    : {acc_sau:.4f}")
if acc_la > acc_sau:
    print("Nhan xet: Cat tia theo so khach toi thieu o la tot hon tren bo du lieu nay.")
elif acc_sau > acc_la:
    print("Nhan xet: Cat tia theo do sau 5 tot hon tren bo du lieu nay.")
else:
    print("Nhan xet: Hai cach cat tia cho cung accuracy tren bo du lieu nay.")
