"""Bước 5: thay đổi ngưỡng phân loại xác suất."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("sinh_vien.csv")
X, y = df[["gio_on"]], df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
for threshold in [0.3, 0.5, 0.7]:
    pred = (prob >= threshold).astype(int)
    print(f"\nNgưỡng {threshold:.1f} | accuracy={accuracy_score(y_test, pred):.3f}")
    print(confusion_matrix(y_test, pred))
