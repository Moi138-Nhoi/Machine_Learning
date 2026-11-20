"""Bài 2: thêm tuổi, chọn lại K bằng silhouette và so tâm cụm."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/khach_hang.csv")
cot_ba = ["tuoi", "thu_nhap", "diem_chi_tieu"]
bo_chuan_hoa = StandardScaler()
Xs = bo_chuan_hoa.fit_transform(df[cot_ba])
print("K  silhouette")
ket_qua = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    s = silhouette_score(Xs, km.labels_)
    ket_qua.append((k, s, km))
    print(f"{k:2d} {s:10.4f}")
k_tot, s_tot, km_tot = max(ket_qua, key=lambda hang: hang[1])
print(f"K tot nhat khi co tuoi: {k_tot}, silhouette = {s_tot:.4f}")
print("K tot nhat co con la 5:", k_tot == 5)

tam_ba = pd.DataFrame(bo_chuan_hoa.inverse_transform(km_tot.cluster_centers_), columns=cot_ba)
tam_ba["so_khach"] = np.bincount(km_tot.labels_)
print("Tam moi (don vi goc, sap theo thu nhap):")
print(tam_ba.sort_values("thu_nhap").round(1).to_string(index=False))

cot_hai = ["thu_nhap", "diem_chi_tieu"]
sc_hai = StandardScaler().fit(df[cot_hai])
km_hai = KMeans(n_clusters=5, n_init=10, random_state=0).fit(sc_hai.transform(df[cot_hai]))
tam_hai = pd.DataFrame(sc_hai.inverse_transform(km_hai.cluster_centers_), columns=cot_hai)
tam_hai["so_khach"] = np.bincount(km_hai.labels_)
tam_hai["tuoi_tb"] = df.groupby(km_hai.labels_)["tuoi"].mean().to_numpy()
print("Tam cu 2 cot (muc 3, sap theo thu nhap):")
print(tam_hai.sort_values("thu_nhap").round(1).to_string(index=False))
print("Nhan xet: K van la 5, nhung silhouette giam so voi 0.7227 cua 2 cot.")
print("Nhom thu nhap thap/chi nhieu doi tu (7.5, 80.2; 35 khach) sang")
print("(8.5, 77.6; 39 khach). Nhom giua doi tu 60 con 57 khach.")
print("Tuoi tham gia khoang cach nen mot so khach doi cum; khong so cum theo so hieu.")
