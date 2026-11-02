"""Tính Gini bằng tay và tìm câu hỏi chia tốt nhất cho một nút."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def gini(nhan):
    """Chỉ số Gini của một nút phân loại nhị phân."""
    nhan = np.asarray(nhan)
    if len(nhan) == 0:
        return 0.0
    p1 = nhan.mean()
    return 1 - p1**2 - (1 - p1) ** 2


def gini_sau_chia(nhan, di_trai):
    """Trung bình Gini của hai nhánh, có trọng số theo số mẫu."""
    nhan = np.asarray(nhan)
    di_trai = np.asarray(di_trai, dtype=bool)
    trai, phai = nhan[di_trai], nhan[~di_trai]
    return (len(trai) * gini(trai) + len(phai) * gini(phai)) / len(nhan)


def tim_cau_hoi(X, y):
    """Trả về (tên cột, ngưỡng) có Gini sau chia thấp nhất.

    Thử trung điểm của mọi cặp giá trị liền nhau trong từng cột số.
    Chỉ xét hai nhánh đều có mẫu; nếu các cột đều là hằng số thì báo lỗi.
    """
    nhan = np.asarray(y)
    if len(X) != len(nhan):
        raise ValueError("X va y phai co cung so dong")
    cot_tot, nguong_tot, gini_tot = None, None, float("inf")
    for ten_cot in X.columns:
        gia_tri = X[ten_cot].to_numpy()
        muc = np.sort(np.unique(gia_tri))
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (a + b) / 2
            g = gini_sau_chia(nhan, gia_tri <= nguong)
            if g < gini_tot:
                cot_tot, nguong_tot, gini_tot = ten_cot, float(nguong), g
    if cot_tot is None:
        raise ValueError("Khong co cot nao co the chia")
    return cot_tot, nguong_tot


def main():
    # Mười khách vay giống bảng trong tài liệu.
    nho = pd.DataFrame({
        "thu_nhap": [6, 8, 9, 11, 13, 15, 18, 20, 25, 30],
        "co_viec_on_dinh": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "tra_dung_han": [0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    })
    y = nho["tra_dung_han"].to_numpy()
    g_goc = gini(y)
    print(f"Gini cua nut goc: {g_goc:.4f}\n")

    thu = nho["thu_nhap"].to_numpy()
    muc = np.sort(np.unique(thu))
    print("Cau hoi 'thu_nhap <= nguong ?'")
    print(" nguong Gini sau chia Gini giam")
    for a, b in zip(muc[:-1], muc[1:]):
        nguong = (a + b) / 2
        g = gini_sau_chia(y, thu <= nguong)
        print(f" {nguong:6.1f} {g:13.4f} {g_goc - g:9.4f}")

    viec = nho["co_viec_on_dinh"].to_numpy()
    g_viec = gini_sau_chia(y, viec == 0)
    print("\nCau hoi 'co viec on dinh ?'")
    print(f" Gini sau chia {g_viec:.4f}, Gini giam {g_goc - g_viec:.4f}\n")

    cot = ["thu_nhap", "co_viec_on_dinh"]
    ten_cot, nguong = tim_cau_hoi(nho[cot], y)
    print(f"Tu tinh: {ten_cot} <= {nguong:.2f}")
    cay = DecisionTreeClassifier(max_depth=1, random_state=0).fit(nho[cot], y)
    print("Cay mot tang do scikit-learn chon:")
    print(export_text(cay, feature_names=cot, show_weights=True))


if __name__ == "__main__":
    main()
