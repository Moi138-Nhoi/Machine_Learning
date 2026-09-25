
import numpy as np
import pandas as pd

df = pd.read_csv("gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Chuẩn hóa diện tích
x_tb = x_goc.mean()
x_std = x_goc.std()

x = (x_goc - x_tb) / x_std


def chay_gradient(toc_do_hoc):
    w = 0.0
    b = 0.0

    so_vong = 200

    for vong in range(so_vong):
        y_du_doan = w * x + b

        sai_so = y_du_doan - y

        dw = (2 / len(x)) * (sai_so * x).sum()
        db = (2 / len(x)) * sai_so.sum()

        w = w - toc_do_hoc * dw
        b = b - toc_do_hoc * db

    y_du_doan = w * x + b
    mse = ((y - y_du_doan) ** 2).mean()

    return mse


mse_001 = chay_gradient(0.001)
mse_102 = chay_gradient(1.02)

print(f"Toc do hoc = 0.001 -> MSE sau 200 vong = {mse_001:.4f}")
print(f"Toc do hoc = 1.02  -> MSE sau 200 vong = {mse_102:.4f}")

print()
print("Nhan xet:")

print(
    "Toc do hoc 0.001 rat nho nen moi buoc cap nhat rat ngan, "
    "vi vay sau 200 vong mo hinh chua tien den diem toi uu."
)

print(
    "Toc do hoc 1.02 qua lon nen mo hinh buoc qua diem toi uu, "
    "lam sai so tang rat manh thay vi giam dan."
)