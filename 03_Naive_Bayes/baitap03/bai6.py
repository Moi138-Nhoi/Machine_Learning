# -*- coding: utf-8 -*-
"""Bài 6: ảnh hưởng của lớp hiếm và tiên nghiệm, dựa trên d3."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "tin_nhan.csv")
    # Giữ tất cả tin thường và đúng 20 tin rác đầu tiên theo thứ tự CSV.
    mat_can_bang = pd.concat([
        df[df["la_rac"] == 0], df[df["la_rac"] == 1].head(20),
    ], ignore_index=True)
    print("So tin theo nhan (0: thuong, 1: rac):")
    print(mat_can_bang["la_rac"].value_counts().sort_index().to_string())

    tin_hoc, tin_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        mat_can_bang["noi_dung"], mat_can_bang["la_rac"],
        test_size=0.25, random_state=17, stratify=mat_can_bang["la_rac"],
    )
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_hoc = vec.fit_transform(tin_hoc)
    X_kiem_tra = vec.transform(tin_kiem_tra)
    mo_hinh = MultinomialNB(alpha=1.0).fit(X_hoc, y_hoc)
    du_doan = mo_hinh.predict(X_kiem_tra)

    print(f"\nTap hoc: {len(y_hoc)} tin; tap kiem tra: {len(y_kiem_tra)} tin.")
    print("Tien nghiem hoc duoc (thuong, rac):", np.exp(mo_hinh.class_log_prior_))
    print(f"Accuracy = {accuracy_score(y_kiem_tra, du_doan):.4f}")
    print(f"Recall lop rac = {recall_score(y_kiem_tra, du_doan, pos_label=1, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_kiem_tra, du_doan, labels=[0, 1]).ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Mốc đối chứng cho thấy vì sao chỉ nhìn accuracy có thể bị đánh lừa.
    luon_thuong = np.zeros(len(y_kiem_tra), dtype=int)
    print("\nMoc doi chung: luon doan tin thuong")
    print(f"Accuracy = {accuracy_score(y_kiem_tra, luon_thuong):.4f}")
    print(f"Recall lop rac = {recall_score(y_kiem_tra, luon_thuong, pos_label=1, zero_division=0):.4f}")
    print("\nNhan xet: accuracy khong du de danh gia bo loc khi lop rac hiem;")
    print("chi doan thuong da dat 90% accuracy nhung khong chan duoc tin rac nao.")
    print("Tien nghiem lop thuong la 0.9, lop rac la 0.1; bang chung tu ngu phai")
    print("du manh de vuot su chenh lech nay. Chi 15 tin rac duoc dung de hoc,")
    print("nen uoc luong tu cua lop rac cung kem on dinh hon.")
    print(f"Bo loc bat duoc {tp}/{tp + fn} tin rac; can xem recall cung voi accuracy.")
    print("Tap kiem tra chi co 5 tin rac, nen moi tin rac lam recall doi 0.2.")


if __name__ == "__main__":
    main()
