"""Bài 3: chọn ngưỡng lớn nhất đạt recall >= 0.8 trên tập học."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = make_pipeline(StandardScaler(), LogisticRegression())

# Mỗi xác suất được tạo bởi mô hình chưa từng học trên chính dòng đó.
xs_hoc = cross_val_predict(mo_hinh, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
# Recall chỉ đổi khi ngưỡng đi qua một xác suất đã dự đoán.
nguong_co_the = np.unique(np.r_[0.0, xs_hoc, 1.0])
nguong_dat = [
    t for t in nguong_co_the
    if recall_score(y_tr, (xs_hoc >= t).astype(int)) >= 0.8
]
nguong = max(nguong_dat)
du_hoc = (xs_hoc >= nguong).astype(int)
print(f"Nguong lon nhat chon tren tap hoc: {nguong:.8f}")
print(f"Recall kiem dinh cheo tai nguong nay: {recall_score(y_tr, du_hoc):.4f}")
print(f"Precision kiem dinh cheo: {precision_score(y_tr, du_hoc):.4f}")

# Chỉ sau khi cố định ngưỡng mới chấm đúng một lần trên tập kiểm tra.
mo_hinh.fit(X_tr, y_tr)
xs_kiem_tra = mo_hinh.predict_proba(X_te)[:, 1]
du_kiem_tra = (xs_kiem_tra >= nguong).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, du_kiem_tra).ravel()
recall = recall_score(y_te, du_kiem_tra)
precision = precision_score(y_te, du_kiem_tra)
print(f"Tap kiem tra: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Recall={recall:.4f}; precision={precision:.4f}; so ban bi bao={tp + fp}")
print("Dat yeu cau recall >= 0.8 tren tap kiem tra:", recall >= 0.8)
print(
    "Giai thich: nguong duoc chon tu du doan kiem dinh cheo tren tap hoc. "
    "Recall tap kiem tra co the khac 0.8 do mau kiem tra khac; "
    "khong duoc doi nguong sau khi xem ket qua tap kiem tra."
)
