# -*- coding: utf-8 -*-
"""Bài 4: so sánh bỏ phiếu đều và theo khoảng cách trên hai vầng trăng."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X, y = df[["x1", "x2"]], df["nhan"]
    X_hoc, X_kt, y_hoc, y_kt = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y)
    ket_qua = {}
    print(f"{'k':>4s} {'weights':12s} {'Acc hoc':>12s} {'Acc kiem tra':>14s}")
    for k in [1, 13, 101]:
        for weights in ["uniform", "distance"]:
            mh = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X_hoc, y_hoc)
            acc = accuracy_score(y_kt, mh.predict(X_kt))
            ket_qua[k, weights] = acc
            print(f"{k:4d} {weights:12s} {accuracy_score(y_hoc, mh.predict(X_hoc)):12.4f} {acc:14.4f}")
    chenh = ket_qua[101, 'distance'] - ket_qua[101, 'uniform']
    print(f"\nk=101: accuracy distance - uniform = {chenh:+.4f}.")
    print("Voi k=101, distance giup mo hinh bot cau tha: accuracy tang 5.33 diem %.")
    print("Vi phieu 1/d uu tien mau gan, giam anh huong mau xa va giu cau truc cuc bo.")
    print("Voi k=1, chi co mot lang gieng nen hai cach bo phieu giong nhau.")
    print("Accuracy tap hoc cua distance co the bang 1 vi moi diem hoc cach")
    print("chinh no bang 0; can dung diem kiem tra de danh gia.")


if __name__ == "__main__":
    main()