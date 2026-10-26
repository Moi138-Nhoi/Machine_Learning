# -*- coding: utf-8 -*-
"""Bài 5: chọn k hồi quy bằng MAE kiểm định chéo, so với đường thẳng e5."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/gia_nha.csv")
    X, y = df[["dien_tich"]], df["gia"]
    X_hoc, X_kt, y_hoc, y_kt = train_test_split(X, y, test_size=0.2, random_state=42)
    diem = {}
    print(f"{'k':>4s} {'MAE CV (ty dong)':>20s}")
    for k in range(1, 21):
        diem[k] = cross_val_score(KNeighborsRegressor(n_neighbors=k), X_hoc, y_hoc,
                                 cv=5, scoring="neg_mean_absolute_error").mean()
        print(f"{k:4d} {-diem[k]:20.4f}")
    # Điểm âm càng lớn càng tốt, tương ứng MAE càng nhỏ.
    k_tot = max(diem, key=diem.get)
    knn = KNeighborsRegressor(n_neighbors=k_tot).fit(X_hoc, y_hoc)
    duong = LinearRegression().fit(X_hoc, y_hoc)
    mae_knn = mean_absolute_error(y_kt, knn.predict(X_kt))
    mae_duong = mean_absolute_error(y_kt, duong.predict(X_kt))
    print(f"\nk tot nhat theo CV = {k_tot}; MAE CV = {-diem[k_tot]:.4f}")
    print(f"MAE kiem tra KNN: {mae_knn:.4f} ty dong")
    print(f"MAE kiem tra duong thang: {mae_duong:.4f} ty dong")
    print("KNN " + ("vuot" if mae_knn < mae_duong else "khong vuot") + " duoc duong thang tren tap kiem tra.")
    print("Gia nha tang gan tuyen tinh theo dien tich nen duong thang la moc")
    print("doi chung phu hop; chon k bang CV khong dam bao KNN se thang.")


if __name__ == "__main__":
    main()