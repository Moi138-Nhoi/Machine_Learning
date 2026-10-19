# -*- coding: utf-8 -*-
"""Bài 3: so sánh hệ số làm trơn trên cùng một lần chia dữ liệu, dựa trên d3."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "tin_nhan.csv")
    tin_hoc, tin_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        df["noi_dung"], df["la_rac"], test_size=0.25,
        random_state=17, stratify=df["la_rac"],
    )
    # Chỉ lập từ điển từ tập học để không lấy thông tin của tập kiểm tra.
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_hoc = vec.fit_transform(tin_hoc)
    X_kiem_tra = vec.transform(tin_kiem_tra)

    print(f"Tap hoc: {len(y_hoc)} tin; tap kiem tra: {len(y_kiem_tra)} tin.")
    print(f"{'alpha':>8s} {'accuracy':>12s} {'precision':>12s} {'recall':>12s}")
    for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
        mo_hinh = MultinomialNB(alpha=alpha).fit(X_hoc, y_hoc)
        du_doan = mo_hinh.predict(X_kiem_tra)
        print(f"{alpha:8.3f} {accuracy_score(y_kiem_tra, du_doan):12.4f} "
              f"{precision_score(y_kiem_tra, du_doan, zero_division=0):12.4f} "
              f"{recall_score(y_kiem_tra, du_doan, zero_division=0):12.4f}")

    print("\nNhan xet: alpha qua lon lam P(tu|lop) tien gan 1/V, lam mo su khac")
    print("biet giua hai lop; tien nghiem chi phoi hon va mo hinh de nghieng ve")
    print("lop thuong dong hon, nen co the bo sot tin rac (recall giam).")
    print("Tren tap nho nay, cac alpha co the cho cung chi so du xac suat da doi.")


if __name__ == "__main__":
    main()
