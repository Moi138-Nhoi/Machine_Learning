"""Bài 6: phân cụm chữ số không dùng nhãn khi học, sau đó đối chiếu nhãn thật."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


X, nhan_that = load_digits(return_X_y=True)
cum = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(X)
bang = pd.crosstab(
    pd.Series(nhan_that, name="nhan_that"), pd.Series(cum, name="cum")
)
print("Bang nhan that x cum:")
print(bang.to_string())
print()
ty_le_tap_trung = bang.max(axis=1) / bang.sum(axis=1)
print("Ty le cua moi chu so nam trong cum dong nhat:")
for so, ty_le in ty_le_tap_trung.items():
    cum_chinh = int(bang.loc[so].idxmax())
    print(f"  So {so}: {ty_le:.3f} trong cum {cum_chinh} "
          f"({bang.loc[so, cum_chinh]}/{bang.loc[so].sum()} anh)")
so_gon = int(ty_le_tap_trung.idxmax())
print(f"Chu so duoc gom gon nhat: {so_gon}")
print("Cac cap chu so cung xuat hien nhieu trong mot cum:")
for ma_cum in bang.columns:
    hai_so = bang[ma_cum].sort_values(ascending=False).head(2)
    if hai_so.iloc[1] >= 10:
        print(f"  Cum {ma_cum}: so {hai_so.index[0]} ({hai_so.iloc[0]}) "
              f"va so {hai_so.index[1]} ({hai_so.iloc[1]})")
print("Nhan xet: K-Means nhom theo do gan pixel, khong biet nhan chu so;")
print("mot chu so co the bi tach ra nhieu cum va mot cum co the chua nhieu so.")
