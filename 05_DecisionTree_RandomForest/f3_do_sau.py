"""So sánh cây tự do và cây chọn độ sâu bằng kiểm định chéo."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(Path(__file__).parent / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

tu_do = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("Cay moc tu do")
print(f" do sau {tu_do.get_depth()}, so la {tu_do.get_n_leaves()}")
print(f" tap hoc {accuracy_score(y_tr, tu_do.predict(X_tr)):.4f}, "
      f"tap kiem tra {accuracy_score(y_te, tu_do.predict(X_te)):.4f}\n")

diem = {}
print("Do sau  tap hoc  kiem dinh cheo")
for do_sau in range(1, 16):
    mh = DecisionTreeClassifier(max_depth=do_sau, random_state=0)
    diem[do_sau] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    hoc = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))
    print(f" {do_sau:2d}       {hoc:.4f}   {diem[do_sau]:.4f}")

d_tot = max(diem, key=diem.get)
tot = DecisionTreeClassifier(max_depth=d_tot, random_state=0).fit(X_tr, y_tr)
print(f"\nDo sau tot nhat = {d_tot}, cay co {tot.get_n_leaves()} la")
print(f"Accuracy tren tap kiem tra = {accuracy_score(y_te, tot.predict(X_te)):.4f}")
