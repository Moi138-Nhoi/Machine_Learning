"""Bài 6: đo thời gian fit của SVC và LinearSVC theo cỡ dữ liệu."""

from time import perf_counter

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC


kich_thuoc = [2000, 5000, 10000, 20000]
ket_qua = []
print("      n    SVC (s)   LinearSVC (s)   SVC/lan truoc   Linear/lan truoc")
for n in kich_thuoc:
    X, y = make_classification(n_samples=n, n_features=20, random_state=0)
    thoi_gian = []
    for mo_hinh in [SVC(), LinearSVC(dual=False, max_iter=5000)]:
        bat_dau = perf_counter()
        mo_hinh.fit(X, y)
        thoi_gian.append(perf_counter() - bat_dau)
    if ket_qua:
        ty_le_svc = f"{thoi_gian[0] / ket_qua[-1][1]:.2f}x"
        ty_le_linear = f"{thoi_gian[1] / ket_qua[-1][2]:.2f}x"
    else:
        ty_le_svc = ty_le_linear = "-"
    print(
        f"{n:7d}  {thoi_gian[0]:9.4f}  {thoi_gian[1]:14.4f}"
        f"  {ty_le_svc:>13s}  {ty_le_linear:>16s}"
    )
    ket_qua.append((n, *thoi_gian))

svc_gap_doi = ket_qua[-1][1] / ket_qua[-2][1]
linear_gap_doi = ket_qua[-1][2] / ket_qua[-2][2]
print(
    f"Nhan xet khi n tang tu 10000 len 20000: SVC cham hon "
    f"{svc_gap_doi:.2f} lan; LinearSVC cham hon {linear_gap_doi:.2f} lan."
)
print(
    "SVC dung kernel RBF nen thuong tang thoi gian nhanh hon LinearSVC; "
    "cac so do phu thuoc may va co dao dong."
)
