"""Bài 4: chia dữ liệu, huấn luyện và đánh giá mô hình."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("sinh_vien.csv")
X, y = df[["gio_on"]], df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Số mẫu train/test:", len(X_train), len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
