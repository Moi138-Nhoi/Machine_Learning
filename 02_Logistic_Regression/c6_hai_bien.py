"""Bước 6: hồi quy logistic với hai đặc trưng đầu vào."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("sinh_vien.csv")
features = ["gio_on", "diem_giua_ky"]
X, y = df[features], df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)
print("Hệ số theo [gio_on, diem_giua_ky]:", model.coef_[0])
print("Hệ số chặn:", model.intercept_[0])
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
