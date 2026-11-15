"""Bài 6: một dự đoán hỏng ảnh hưởng MAE, RMSE và R2 thế nào."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]  # Cùng cách học hồi quy tuyến tính ở bài 1.
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)
du_doan = LinearRegression().fit(X_tr, y_tr).predict(X_te)


def cham(du):
    return (
        mean_absolute_error(y_te, du),
        np.sqrt(mean_squared_error(y_te, du)),
        r2_score(y_te, du),
    )


ban_dau = cham(du_doan)
du_hong = du_doan.copy()
du_hong[0] += 5.0  # Đơn vị giá nhà trong CSV là tỷ đồng.
sau_khi_hong = cham(du_hong)
print(f"So can kiem tra: {len(y_te)}")
print(f"Can dau tien: gia that = {y_te.iloc[0]:.4f}; du doan goc = {du_doan[0]:.4f}; "
      f"du doan hỏng = {du_hong[0]:.4f} ty dong")
print("                  MAE       RMSE         R2")
print(f"Ban dau       {ban_dau[0]:9.4f} {ban_dau[1]:9.4f} {ban_dau[2]:10.4f}")
print(f"Sau cong 5 ty {sau_khi_hong[0]:9.4f} {sau_khi_hong[1]:9.4f} {sau_khi_hong[2]:10.4f}")
print(f"Do thay doi    {sau_khi_hong[0]-ban_dau[0]:+9.4f} "
      f"{sau_khi_hong[1]-ban_dau[1]:+9.4f} {sau_khi_hong[2]-ban_dau[2]:+10.4f}")
print(
    "Nhan xet: RMSE tang manh hon MAE vi sai so moi bi binh phuong "
    "truoc khi lay trung binh. R2 giam vi tong binh phuong sai so tang. "
    "MAE/RMSE co don vi ty dong, con R2 khong co don vi nen khong nen "
    "so sanh truc tiep do lech so hoc cua R2 voi hai so kia."
)
