"""Minh họa bootstrap và so sánh rừng ngẫu nhiên với cây đơn."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(Path(__file__).parent / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

rng = np.random.default_rng(0)
n = len(X_tr)
chi_so = rng.integers(0, n, n)
khac = len(np.unique(chi_so))
print(f"Boc {n} lan, co {khac} khach khac nhau ({khac / n:.0%})\n")

rung = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
print("Rung 200 cay")
print(f" tap hoc       : {accuracy_score(y_tr, rung.predict(X_tr)):.4f}")
print(f" tap kiem tra  : {accuracy_score(y_te, rung.predict(X_te)):.4f}\n")

print("So cay  accuracy tap kiem tra")
for so_cay in [1, 10, 50, 200, 400]:
    mh = RandomForestClassifier(n_estimators=so_cay, random_state=0).fit(X_tr, y_tr)
    print(f" {so_cay:4d}   {accuracy_score(y_te, mh.predict(X_te)):.4f}")

ba_mo_hinh = {
    "Cay moc tu do": DecisionTreeClassifier(random_state=0),
    "Cay sau 5": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Rung 200 cay": rung,
}
print("\nMo hinh         tap hoc  tap kiem tra")
for ten, mh in ba_mo_hinh.items():
    mh.fit(X_tr, y_tr)
    a = accuracy_score(y_tr, mh.predict(X_tr))
    b = accuracy_score(y_te, mh.predict(X_te))
    print(f" {ten:14s} {a:.4f}    {b:.4f}")
