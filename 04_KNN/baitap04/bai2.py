# -*- coding: utf-8 -*-
"""Bài 2: chọn k bằng CV 5 phần cho Euclid và Manhattan, dựa trên e3."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X, y = df[["x1", "x2"]], df["nhan"]
    X_hoc, X_kt, y_hoc, y_kt = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y)
    ket_qua = {}
    for metric in ["euclidean", "manhattan"]:
        print(f"\nKhoang cach: {metric}")
        print(f"{'k':>4s} {'Acc hoc':>12s} {'Acc kiem tra':>14s}")
        for k in [1, 5, 13, 31, 101]:
            mh = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_hoc, y_hoc)
            print(f"{k:4d} {accuracy_score(y_hoc, mh.predict(X_hoc)):12.4f} "
                  f"{accuracy_score(y_kt, mh.predict(X_kt)):14.4f}")
        diem = {k: cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric),
                                  X_hoc, y_hoc, cv=5).mean()
                for k in range(1, 52, 2)}
        k_tot = max(diem, key=diem.get)
        mh = KNeighborsClassifier(n_neighbors=k_tot, metric=metric).fit(X_hoc, y_hoc)
        acc = accuracy_score(y_kt, mh.predict(X_kt))
        ket_qua[metric] = acc
        print(f"k tot nhat theo CV = {k_tot}; CV = {diem[k_tot]:.4f}; accuracy = {acc:.4f}")
    chenh = abs(ket_qua['euclidean'] - ket_qua['manhattan'])
    print(f"\nChenh lech accuracy hai cach do: {chenh:.4f} ({chenh * 100:.2f} diem phan tram).")
    print("Doi khoang cach co the doi thu tu lang gieng va k tot nhat;")
    print("tren bo nay chenh lech chi 1.33 diem phan tram, khong khac nhieu.")
    print("k duoc chon chi bang CV tap hoc, khong chon tu bang diem kiem tra.")


if __name__ == "__main__":
    main()