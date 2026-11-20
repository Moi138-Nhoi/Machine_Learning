"""Bài 3: so 50 lần khởi tạo ngẫu nhiên và k-means++."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
so_lan_dat = {}
print("Cach khoi tao  So lan inertia lam tron = 23.0 / 50  Min    Max    Trung binh")
for cach in ["random", "k-means++"]:
    inertia = np.array([
        KMeans(n_clusters=5, init=cach, n_init=1, random_state=rs).fit(Xs).inertia_
        for rs in range(50)
    ])
    so_lan_dat[cach] = int(np.sum(np.round(inertia, 1) == 23.0))
    print(f"{cach:12s} {so_lan_dat[cach]:3d}/50"
          f"                          {inertia.min():5.1f} {inertia.max():6.1f} {inertia.mean():10.1f}")
    print("  Inertia (>23.05) o cac lan khong dat:", np.round(inertia[inertia > 23.05], 1).tolist())

tot_hon = max(so_lan_dat, key=so_lan_dat.get)
print("Cach dang tin hon theo so lan dat 23.0:", tot_hon)
print("Giai thich: k-means++ chon cac tam ban dau cach xa nhau, giam nguy co")
print("ket thuc tai nghiem cuc bo te; khoi tao random 1 lan de bi ket qua phu thuoc seed.")
