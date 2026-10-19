# -*- coding: utf-8 -*-
"""Bài 4: tự đếm từ, làm trơn và cộng loga, mở rộng từ d4.

Bộ lọc tự viết chỉ dùng thư viện chuẩn Python. Scikit-learn được dùng để
chia dữ liệu, chấm điểm và xây mô hình đối chứng theo yêu cầu đề.
"""

from collections import Counter
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def tach_tu(cau):
    # Giống CountVectorizer(lowercase=True, token_pattern=r"(?u)\S+").
    return cau.lower().split()


class BoLocTuViet:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, cac_tin, cac_nhan):
        self.dem_tu = {0: Counter(), 1: Counter()}
        self.dem_tin = Counter()
        self.tu_dien = set()
        for tin, nhan in zip(cac_tin, cac_nhan):
            cac_tu = tach_tu(tin)
            self.dem_tu[nhan].update(cac_tu)
            self.dem_tin[nhan] += 1
            self.tu_dien.update(cac_tu)

        so_tin = sum(self.dem_tin.values())
        V = len(self.tu_dien)
        self.log_tien_nghiem = {}
        self.log_xac_suat = {}
        for nhan in [0, 1]:
            self.log_tien_nghiem[nhan] = log(self.dem_tin[nhan] / so_tin)
            tong_tu = sum(self.dem_tu[nhan].values())
            self.log_xac_suat[nhan] = {
                tu: log((self.dem_tu[nhan][tu] + self.alpha)
                        / (tong_tu + self.alpha * V))
                for tu in self.tu_dien
            }
        return self

    def tinh_diem(self, tin):
        dem = Counter(tach_tu(tin))
        return [
            self.log_tien_nghiem[nhan] + sum(
                so_lan * self.log_xac_suat[nhan][tu]
                for tu, so_lan in dem.items() if tu in self.tu_dien
            )
            for nhan in [0, 1]
        ]

    def predict(self, cac_tin):
        ket_qua = []
        for tin in cac_tin:
            diem_thuong, diem_rac = self.tinh_diem(tin)
            # Khi hòa, chọn lớp 0 giống cách argmax của MultinomialNB.
            ket_qua.append(1 if diem_rac > diem_thuong else 0)
        return ket_qua


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "tin_nhan.csv")
    tin_hoc, tin_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        df["noi_dung"], df["la_rac"], test_size=0.25,
        random_state=17, stratify=df["la_rac"],
    )
    tu_viet = BoLocTuViet(alpha=1.0).fit(tin_hoc, y_hoc)
    du_doan_tay = tu_viet.predict(tin_kiem_tra)

    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_hoc = vec.fit_transform(tin_hoc)
    X_kiem_tra = vec.transform(tin_kiem_tra)
    doi_chung = MultinomialNB(alpha=1.0).fit(X_hoc, y_hoc)
    du_doan_thu_vien = doi_chung.predict(X_kiem_tra)

    print(f"Tap hoc: {len(y_hoc)} tin; tap kiem tra: {len(y_kiem_tra)} tin.")
    print(f"Tu dien tap hoc: {len(tu_viet.tu_dien)} tu.")
    print(f"Accuracy tu viet       = {accuracy_score(y_kiem_tra, du_doan_tay):.4f}")
    print(f"Accuracy MultinomialNB = {accuracy_score(y_kiem_tra, du_doan_thu_vien):.4f}")
    so_khac = sum(a != b for a, b in zip(du_doan_tay, du_doan_thu_vien))
    print(f"So du doan khac nhau = {so_khac}")
    print("Ma tran nham lan tu viet (dong: nhan that; cot: du doan; thu tu 0, 1):")
    print(confusion_matrix(y_kiem_tra, du_doan_tay, labels=[0, 1]))

    # Kiểm tra cả điểm số để tránh trường hợp accuracy trùng do tình cờ.
    diem_tay = np.array([tu_viet.tinh_diem(tin) for tin in tin_kiem_tra])
    diem_doi_chung = (X_kiem_tra @ doi_chung.feature_log_prob_.T
                     + doi_chung.class_log_prior_)
    sai_so = np.max(np.abs(diem_tay - diem_doi_chung))
    print(f"Sai so diem log lon nhat = {sai_so:.3e}")
    assert np.allclose(diem_tay, diem_doi_chung)
    assert so_khac == 0

    print("\nGiai thich: hai bo loc cung dem so LAN xuat hien cua tu, cung tu dien")
    print("tap hoc, alpha=1 va tien nghiem bang ty le tin cua moi lop, nen khop nhau.")
    print("Cong loga tranh tran duoi; tu lap lai duoc tinh theo so lan xuat hien.")
    print("Tu chua co trong tu dien tap hoc bi bo qua, giong CountVectorizer.")
    print("Neu lech, can kiem tra tach tu, tong so tu moi lop va cach chia du lieu.")


if __name__ == "__main__":
    main()
