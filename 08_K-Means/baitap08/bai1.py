"""Bài 1: tự tính silhouette từng khách hàng chỉ bằng NumPy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Trả về (a, b, s) theo đúng định nghĩa silhouette."""
    X = np.asarray(X)
    nhan = np.asarray(nhan)
    kc = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
    cung_cum = (nhan == nhan[i])
    cung_cum[i] = False  # Không tính khoảng cách tới chính mình vào a.
    if not np.any(cung_cum):
        return 0.0, min(np.mean(kc[nhan == c]) for c in np.unique(nhan) if c != nhan[i]), 0.0
    a = np.mean(kc[cung_cum])
    b = min(np.mean(kc[nhan == c]) for c in np.unique(nhan) if c != nhan[i])
    s = (b - a) / max(a, b) if max(a, b) else 0.0
    return a, b, s


df = pd.read_csv("data/khach_hang.csv")
cot = ["thu_nhap", "diem_chi_tieu"]
bo_chuan_hoa = StandardScaler()
Xs = bo_chuan_hoa.fit_transform(df[cot])
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
ket_qua = np.array([silhouette_mot_diem(Xs, km.labels_, i) for i in range(len(Xs))])
s_tb = ket_qua[:, 2].mean()
s_sklearn = silhouette_score(Xs, km.labels_)
i_thap = int(np.argmin(ket_qua[:, 2]))
print(f"Silhouette tu tinh: {s_tb:.6f}")
print(f"silhouette_score : {s_sklearn:.6f}")
print(f"Sai lech          : {abs(s_tb - s_sklearn):.10f}")
print(f"Khach co s nho nhat: hang {i_thap} (so thu tu {i_thap + 1})")
print(df.iloc[i_thap].to_string())
print(f"Cum {km.labels_[i_thap]}; a={ket_qua[i_thap, 0]:.4f}, "
      f"b={ket_qua[i_thap, 1]:.4f}, s={ket_qua[i_thap, 2]:.4f}")

tam = bo_chuan_hoa.inverse_transform(km.cluster_centers_)
plt.figure(figsize=(7, 5))
plt.scatter(df["thu_nhap"], df["diem_chi_tieu"], c=km.labels_, cmap="tab10", s=18)
plt.scatter(tam[:, 0], tam[:, 1], c="black", marker="x", s=110, label="Tam cum")
plt.scatter(df.iloc[i_thap]["thu_nhap"], df.iloc[i_thap]["diem_chi_tieu"],
            facecolors="none", edgecolors="red", s=220, linewidths=2,
            label="Khach silhouette thap nhat")
plt.xlabel("Thu nhap (trieu dong)")
plt.ylabel("Diem chi tieu")
plt.title("Vi tri khach hang co silhouette thap nhat")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "khach_silhouette_thap.png", dpi=170)
plt.close()
print("Da luu khach_silhouette_thap.png de xem vi tri tren hinh 5 cum.")
print("Nhan xet: tren Hinh 3, khach o (thu nhap 25.4, chi tieu 41),")
print("giua nhom thu nhap vua/chi vua va nhom thu nhap cao/chi it.")
print("s gan 0 vi khoang cach trung binh toi cum minh (a) gan cum lan can (b).")
