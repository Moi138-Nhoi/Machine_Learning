"""Bài 1: tính f(x) và khoảng cách tới ranh giới SVM bằng phép tính tay."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


X, y = make_blobs(
    n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
)
y = 1 - y
svm = SVC(kernel="linear", C=1000).fit(X, y)
w1, w2 = svm.coef_[0]
b = svm.intercept_[0]
do_dai_w = np.sqrt(w1**2 + w2**2)
do_rong_le = 2 / do_dai_w


def tinh_tay(diem):
    """Chỉ dùng phép nhân, cộng; không gọi decision_function."""
    return w1 * diem[0] + w2 * diem[1] + b


print(f"w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}")
print(f"||w|| = {do_dai_w:.6f}; do rong le = 2/||w|| = {do_rong_le:.6f}")
print("Hai diem moi:")
for diem in [(3.0, 3.0), (5.0, 4.5)]:
    f = tinh_tay(diem)
    khoang_cach = abs(f) / do_dai_w
    print(
        f"  {diem}: f = ({w1:.6f})*{diem[0]} + ({w2:.6f})*{diem[1]} "
        f"+ ({b:.6f}) = {f:+.6f}; d = |f|/||w|| = {khoang_cach:.6f}; "
        f"lop {1 if f > 0 else 0}"
    )

print("Ba vecto ho tro:")
for chi_so in svm.support_:
    f = tinh_tay(X[chi_so])
    khoang_cach = abs(f) / do_dai_w
    print(
        f"  diem {chi_so:2d}, lop {y[chi_so]}: "
        f"f = {f:+.6f}; d = {khoang_cach:.6f}"
    )

print(f"Nua do rong le = 1/||w|| = {do_rong_le / 2:.6f}")
print(
    "Nhan xet: vecto ho tro nam tren hai duong f(x) = +1 va f(x) = -1. "
    "Vi vay khoang cach toi f(x) = 0 la 1/||w||, bang nua do rong le 2/||w||. "
    "Sai lech rat nho trong so tinh duoc la sai so toi uu so."
)
