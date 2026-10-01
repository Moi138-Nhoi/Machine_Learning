"""Bài 3: huấn luyện logistic regression từ gio_on và dự đoán."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sinh_vien.csv")
model = LogisticRegression().fit(df[["gio_on"]], df["qua_mon"])
gio_moi = pd.DataFrame({"gio_on": [8, 12, 16, 20, 24]})
gio_moi["xac_suat_qua"] = model.predict_proba(gio_moi)[:, 1]
gio_moi["du_doan"] = model.predict(gio_moi[["gio_on"]])
print(gio_moi.to_string(index=False))
