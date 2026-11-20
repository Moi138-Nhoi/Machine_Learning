"""Bài 4: lưu ảnh gốc và ảnh nén 16 màu, kể cả ảnh riêng đọc bằng plt.imread."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


THU_MUC = Path(__file__).resolve().parent


def anh_rgb_float(anh):
    """Đưa ảnh uint8/float, có hoặc không có alpha, về RGB trong [0, 1]."""
    anh = np.asarray(anh)
    if anh.ndim != 3 or anh.shape[2] not in (3, 4):
        raise ValueError("Anh phai co 3 kenh RGB hoac 4 kenh RGBA")
    anh = anh[:, :, :3]
    if np.issubdtype(anh.dtype, np.integer):
        anh = anh.astype(np.float32) / np.iinfo(anh.dtype).max
    else:
        anh = np.clip(anh.astype(np.float32), 0, 1)
    return anh


def nen_va_luu(anh, ten_goc, ten_nen):
    anh = anh_rgb_float(anh)
    cao, rong, _ = anh.shape
    diem = anh.reshape(-1, 3)
    plt.imsave(THU_MUC / ten_goc, anh)
    rng = np.random.default_rng(0)
    mau = diem[rng.choice(len(diem), min(5000, len(diem)), replace=False)]
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    anh_moi = km.cluster_centers_[nhan].reshape(cao, rong, 3)
    plt.imsave(THU_MUC / ten_nen, anh_moi)
    sai_lech = np.abs(anh_moi - anh).mean() * 255
    goc_byte = (THU_MUC / ten_goc).stat().st_size
    nen_byte = (THU_MUC / ten_nen).stat().st_size
    print(f"{ten_goc}: {cao}x{rong}, {goc_byte:,} byte")
    print(f"{ten_nen}: {nen_byte:,} byte; ty le goc/nen = {goc_byte/nen_byte:.2f} lan")
    print(f"Sai lech mau trung binh: {sai_lech:.2f}/255")
    print()


print("Anh china.jpg cua scikit-learn:")
nen_va_luu(load_sample_image("china.jpg"), "anh_goc.png", "anh_16_mau.png")

# Ảnh riêng do sinh viên đặt trong baitap08; có thể truyền đường dẫn khác qua argv.
duong_dan_anh_rieng = (
    Path(sys.argv[1]) if len(sys.argv) > 1
    else THU_MUC / "Screenshot 2026-09-09 215508.png"
)
print("Anh rieng doc bang plt.imread:", duong_dan_anh_rieng.name)
nen_va_luu(plt.imread(duong_dan_anh_rieng), "anh_rieng_goc.png", "anh_rieng_16_mau.png")
print("Nhan xet: 4 bit/diem la muc ly thuyet cho 16 chi so mau, so voi 24 bit RGB.")
print("Dung luong PNG thuc te con phu thuoc nen lossless, header va cach luu mau;")
print("plt.imsave luu anh RGB, khong dong goi palette 4 bit/diem.")
