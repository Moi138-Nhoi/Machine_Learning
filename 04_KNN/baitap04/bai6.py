# -*- coding: utf-8 -*-
"""Bài 6: đo riêng thời gian fit và predict của KNN với ba cỡ dữ liệu."""
from time import perf_counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    rng = np.random.default_rng(0)
    mau_moi = rng.random((1000, 2))
    # Làm nóng thư viện bằng một mô hình nhỏ, ngoài vùng đo thời gian.
    KNeighborsClassifier(n_neighbors=7).fit(
        rng.random((100, 2)), rng.integers(0, 2, 100)).predict(mau_moi[:10])
    print("KNN k=7, algorithm=auto, n_jobs=1; 1000 mau moi, 2 dac trung.")
    print("Thoi gian la trung vi cua 3 lan do (mili giay).")
    print(f"{'So mau hoc':>12s} {'fit (ms)':>14s} {'predict (ms)':>16s} {'Tong (ms)':>14s}")
    for so_mau in [1000, 10000, 100000]:
        X = rng.random((so_mau, 2))
        y = rng.integers(0, 2, size=so_mau)
        fit_ms, predict_ms = [], []
        for _ in range(3):
            mh = KNeighborsClassifier(n_neighbors=7, algorithm="auto", n_jobs=1)
            bat_dau = perf_counter()
            mh.fit(X, y)
            fit_ms.append((perf_counter() - bat_dau) * 1000)
            bat_dau = perf_counter()
            mh.predict(mau_moi)
            predict_ms.append((perf_counter() - bat_dau) * 1000)
        f, p = np.median(fit_ms), np.median(predict_ms)
        print(f"{so_mau:12d} {f:14.3f} {p:16.3f} {f+p:14.3f}")
    print("\nKNN la hoc luoi: fit chu yeu luu mau va co the tao chi muc tim kiem,")
    print("khong uoc luong mot ham du doan nhu hoi quy. Khi predict moi tim")
    print("lang gieng va bo phieu, nen chi phi phu thuoc vao du lieu da luu.")
    print("Voi 2 dac trung, algorithm=auto thuong chon cay KD de tim nhanh;")
    print("vi vay predict khong nhat thiet tang tuyen tinh theo so mau, va fit")
    print("van ton thoi gian xay cay. Do nhieu va lay trung vi giup giam nhieu.")
    print("Thoi gian tuy may va tai he thong; khong dung thoi gian de so accuracy.")


if __name__ == "__main__":
    main()