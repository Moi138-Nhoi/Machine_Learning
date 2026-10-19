# -*- coding: utf-8 -*-
"""Bài 5: đối chiếu biến thể sai và đúng trên dữ liệu liên tục, dựa trên d5."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sinh_vien.csv")
    X = df[["gio_on", "diem_giua_ky"]]
    y = df["qua_mon"]
    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y,
    )
    print(f"Tap hoc: {len(y_hoc)} sinh vien; tap kiem tra: {len(y_kiem_tra)} sinh vien.")
    for ten, mo_hinh in [("MultinomialNB", MultinomialNB(alpha=1.0)),
                         ("GaussianNB", GaussianNB())]:
        mo_hinh.fit(X_hoc, y_hoc)
        du_doan = mo_hinh.predict(X_kiem_tra)
        print(f"{ten:14s}: accuracy = {accuracy_score(y_kiem_tra, du_doan):.4f}")

    print("\nChuong trinh khong bao loi vi hai cot deu la so khong am;")
    print("MultinomialNB chap nhan ca gia tri so thuc khong am.")
    print("Tuy nhien gio on va diem giua ky la dai luong lien tuc, khong phai")
    print("so dem cua cac loai su kien trong mo hinh phan phoi da thuc.")
    print("MultinomialNB coi chung nhu tan suat, nen gia dinh thong ke khong phu hop.")
    print("GaussianNB mo ta tung dac trung lien tuc bang trung binh va phuong sai")
    print("trong moi lop. Accuracy van do duoc khong chung minh bien the da chon dung.")


if __name__ == "__main__":
    main()
