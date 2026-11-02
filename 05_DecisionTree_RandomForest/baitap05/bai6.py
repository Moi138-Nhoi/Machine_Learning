"""Bài 6: rừng hồi quy trên giá nhà và giới hạn ngoại suy."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cay = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)
rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)

print("MAE tren tap kiem tra (ty dong)")
for ten, mh in (("Cay sau 3", cay), ("Duong thang", duong), ("Rung 200 cay", rung)):
    print(f" {ten:12s}: {mean_absolute_error(y_te, mh.predict(X_te)):.4f}")

can_moi = pd.DataFrame({"dien_tich": [250.0]})
gia_rung = rung.predict(can_moi)[0]
print(f"\nGia du doan can 250 m2 cua rung: {gia_rung:.4f} ty dong")
print(f"Khoang gia trong tap hoc: {y_tr.min():.2f} - {y_tr.max():.2f} ty dong")
print("Nhan xet: Rung khong ngoai suy theo dien tich. Moi cay dua ra trung binh "
      "gia cac can trong la cuoi; rung lay trung binh cac du doan do, nen khong "
      "the vuot khoang gia da thay trong tap hoc.")
