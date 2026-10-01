"""Bước 3: khớp mô hình hồi quy logistic với một đặc trưng."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sinh_vien.csv")
X, y = df[["gio_on"]], df["qua_mon"]
model = LogisticRegression().fit(X, y)
print("Hệ số w:", model.coef_[0, 0])
print("Hệ số chặn b:", model.intercept_[0])
for gio in [5, 10, 15, 20, 25]:
    p = model.predict_proba(pd.DataFrame({"gio_on": [gio]}))[0, 1]
    print(f"Ôn {gio:2} giờ: xác suất qua môn = {p:.3f}")
