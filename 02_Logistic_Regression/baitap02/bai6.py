"""Bài 6: sử dụng cả số giờ ôn và điểm giữa kỳ để dự đoán."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("sinh_vien.csv")
features = ["gio_on", "diem_giua_ky"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["qua_mon"], test_size=0.2, random_state=42,
    stratify=df["qua_mon"])
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Hệ số [gio_on, diem_giua_ky]:", model.coef_[0])
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
sv_moi = pd.DataFrame({"gio_on": [12], "diem_giua_ky": [6.5]})
print("Xác suất qua môn của mẫu mới:", model.predict_proba(sv_moi)[0, 1])
