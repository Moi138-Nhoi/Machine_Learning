# -*- coding: utf-8 -*-
"""Bài 3: đổi đơn vị chi tiêu và kiểm tra tác dụng chuẩn hóa, dựa trên e2."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien_mo_rong.csv")
    X, y = df[["gio_on", "diem_giua_ky", "chi_tieu"]], df["qua_mon"]
    X_hoc, X_kt, y_hoc, y_kt = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y)
    du_doan_chuan = []
    print(f"{'Don vi chi tieu':20s} {'Chuan hoa':12s} {'Accuracy':>10s}")
    for don_vi, he_so in [("Nghin dong", 1), ("Trieu dong", 1000)]:
        hoc, kt = X_hoc.copy(), X_kt.copy()
        hoc["chi_tieu"] = hoc["chi_tieu"] / he_so
        kt["chi_tieu"] = kt["chi_tieu"] / he_so
        for chuan in [False, True]:
            mh = (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
                  if chuan else KNeighborsClassifier(n_neighbors=7))
            mh.fit(hoc, y_hoc)
            du_doan = mh.predict(kt)
            print(f"{don_vi:20s} {str(chuan):12s} {accuracy_score(y_kt, du_doan):10.4f}")
            if chuan:
                du_doan_chuan.append(du_doan)
    assert np.array_equal(*du_doan_chuan)
    print("\nGiai thich: chia chi_tieu cho 1000 lam dong gop binh phuong khoang")
    print("cach cua cot nay giam 1 trieu lan; thu tu lang gieng vi the thay doi.")
    print("Don vi nghin dong khien chi_tieu lan at hai cot hoc tap, du khong lien quan.")
    print("Sau chuan hoa, (x/1000 - mean/1000)/(std/1000) = (x-mean)/std,")
    print("nen khoang cach va du doan khong doi (tru sai so so hoc rat nho).")
    print("Scaler chi hoc trung binh, do lech chuan tren tap hoc de tranh ro ri.")


if __name__ == "__main__":
    main()