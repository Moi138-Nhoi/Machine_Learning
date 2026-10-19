# -*- coding: utf-8 -*-
"""Bài 2: tìm từ phân biệt hai lớp bằng hiệu log xác suất, dựa trên d2, d3."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "tin_nhan.csv")
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df["noi_dung"])
    mo_hinh = MultinomialNB(alpha=1.0).fit(X, df["la_rac"])
    # Hiệu này bằng log(P(từ | rác) / P(từ | thường)).
    hieu = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]
    bang = pd.DataFrame({"tu": vec.get_feature_names_out(), "hieu_log": hieu})

    print(f"Huan luyen tren {X.shape[0]} tin nhan, {X.shape[1]} tu.")
    print("\n5 tu to cao tin rac manh nhat (hieu log lon nhat):")
    print(bang.nlargest(5, "hieu_log").to_string(index=False, float_format="%.6f"))
    print("\n5 tu cho thay tin thuong manh nhat (hieu log nho nhat):")
    print(bang.nsmallest(5, "hieu_log").to_string(index=False, float_format="%.6f"))
    print("\nHieu duong ung ho lop rac; hieu am ung ho lop thuong.")
    print("Tri tuyet doi cang lon thi tu cang phan biet hai lop manh.")


if __name__ == "__main__":
    main()
