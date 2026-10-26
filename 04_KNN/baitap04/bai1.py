# -*- coding: utf-8 -*-
"""Bài 1: tự viết KNN bằng NumPy, so với scikit-learn."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Lấy k khoảng cách Euclid nhỏ nhất rồi bỏ phiếu đa số."""
    X_hoc, y_hoc = np.asarray(X_hoc), np.asarray(y_hoc)
    if not 1 <= k <= len(X_hoc):
        raise ValueError("k phai tu 1 den so mau hoc")
    khoang_cach = np.sqrt(((X_hoc - np.asarray(x_moi)) ** 2).sum(axis=1))
    gan_nhat = np.argsort(khoang_cach)[:k]
    nhan, so_phieu = np.unique(y_hoc[gan_nhat], return_counts=True)
    # Nếu hòa, chọn nhãn nhỏ nhất, giống scikit-learn.
    return nhan[np.argmax(so_phieu)]


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X, y = df[["x1", "x2"]].to_numpy(), df["nhan"].to_numpy()
    X_hoc, X_kt, y_hoc, y_kt = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y)
    tu_viet = np.array([knn_du_doan(X_hoc, y_hoc, x, 13) for x in X_kt])
    thu_vien = KNeighborsClassifier(n_neighbors=13).fit(X_hoc, y_hoc).predict(X_kt)
    print(f"Accuracy tu viet: {accuracy_score(y_kt, tu_viet):.4f}")
    print(f"Accuracy sklearn: {accuracy_score(y_kt, thu_vien):.4f}")
    print(f"So du doan khac nhau: {np.count_nonzero(tu_viet != thu_vien)}")
    assert np.array_equal(tu_viet, thu_vien)
    print("Giai thich: moi mau moi duoc so khoang cach voi toan bo tap hoc,")
    print("lay 13 mau gan nhat, moi mau mot phieu; nhan nhieu phieu nhat thang.")
    print("Hai cach cung khoang cach, k va du lieu nen cho ket qua giong nhau.")
    print("Khong dung nhan tap kiem tra de tim lang gieng hay bo phieu.")


if __name__ == "__main__":
    main()