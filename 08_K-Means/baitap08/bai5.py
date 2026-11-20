"""Bài 5: DBSCAN trên hai vầng trăng, so với K-Means."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
nhan_that = df["nhan"]
nhan_kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
ari_kmeans = adjusted_rand_score(nhan_that, nhan_kmeans)
khop = np.mean(nhan_kmeans == nhan_that)
print(f"K-Means: 2 cum, ty le khop nhan = {max(khop, 1-khop):.4f}, ARI = {ari_kmeans:.4f}")
print("eps  so cum  so diem nhieu  ARI so voi nhan that")
for eps in [0.2, 0.3, 0.4]:
    nhan = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    so_cum = len(set(nhan) - {-1})
    so_nhieu = int(np.sum(nhan == -1))
    ari = adjusted_rand_score(nhan_that, nhan)
    print(f"{eps:3.1f} {so_cum:7d} {so_nhieu:14d} {ari:22.4f}")
    print(f"  Tot hon K-Means theo ARI: {ari > ari_kmeans}")
print("Nhan xet: ca ba eps deu cho 1 cum, nen khong tach hai vang trang")
print("tot hon K-Means tren bo du lieu nay. Khi eps tang tu 0.2 len 0.4,")
print("so diem nhieu giam; hai vang trang van noi nhau qua cac diem cau noi.")
