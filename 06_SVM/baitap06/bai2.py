"""Bài 2: xem một điểm mới có làm đổi đường phân cách hay không."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


X, y = make_blobs(
    n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
)
y = 1 - y


def hoc_them(diem=None):
    if diem is None:
        du_lieu, nhan = X, y
    else:
        du_lieu = np.vstack([X, diem])
        nhan = np.append(y, 1)
    return SVC(kernel="linear", C=1000).fit(du_lieu, nhan)


goc = hoc_them()
print("Lan hoc                w1        w2         b    so SV   diem moi la SV")
for ten, diem in [
    ("24 diem goc", None),
    ("Them (-1, -1)", (-1.0, -1.0)),
    ("Them (4.5, 4)", (4.5, 4.0)),
]:
    mo_hinh = hoc_them(diem)
    co_diem_moi = diem is not None and len(X) in mo_hinh.support_
    print(
        f"{ten:19s} {mo_hinh.coef_[0, 0]:+8.4f} "
        f"{mo_hinh.coef_[0, 1]:+8.4f} {mo_hinh.intercept_[0]:+9.4f} "
        f"{len(mo_hinh.support_):7d}   {co_diem_moi if diem else '-'}"
    )
    if diem is not None:
        do_lech = max(
            np.max(np.abs(mo_hinh.coef_ - goc.coef_)),
            np.max(np.abs(mo_hinh.intercept_ - goc.intercept_)),
        )
        f_cu = goc.coef_[0] @ np.asarray(diem) + goc.intercept_[0]
        print(f"  f_goc(diem moi) = {f_cu:+.4f}; do lech he so lon nhat = {do_lech:.6f}")
        if do_lech < 0.01:
            print("  Nhan xet: ranh gioi gan nhu khong doi; diem moi khong quyet dinh le.")
        else:
            print("  Nhan xet: ranh gioi doi vi diem moi nam sat/trong le va tro thanh SV.")

print(
    "Giai thich: diem (-1, -1) o xa ve phia lop 1 nen khong la vecto ho tro. "
    "Diem (4.5, 4) vi pham le cua duong cu, duoc dua vao tap vecto ho tro "
    "va lam thay doi duong phan cach."
)
