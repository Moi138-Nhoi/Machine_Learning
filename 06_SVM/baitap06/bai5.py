"""Bài 5: bảng nhầm lẫn và ảnh chữ số SVM dự đoán sai."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Lưu ảnh được cả khi chạy trong terminal không có màn hình.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du_doan = svm.predict(X_te)
bang = confusion_matrix(y_te, du_doan, labels=np.arange(10))

print("Bang nham lan (hang = so that; cot = so doan):")
print("that\\doan " + " ".join(f"{so:3d}" for so in range(10)))
for so_that, hang in enumerate(bang):
    print(f"{so_that:9d} " + " ".join(f"{so_luong:3d}" for so_luong in hang))

print("Cac cap bi nham (that -> doan):")
for so_that in range(10):
    for so_doan in range(10):
        if so_that != so_doan and bang[so_that, so_doan]:
            print(f"  {so_that} -> {so_doan}: {bang[so_that, so_doan]} anh")

thu_muc_anh = Path("baitap06/anh_sai")
thu_muc_anh.mkdir(parents=True, exist_ok=True)
sai = np.flatnonzero(y_te != du_doan)
print(f"So anh doan sai: {len(sai)}/{len(y_te)}")
nhan_xet_anh = {
    (9, 3): "Net ben trai cua so 9 yeu, hai vong tren/duoi trong giong so 3.",
    (8, 1): "Hai vong cua so 8 bi ho, net doc giua noi bat nhu so 1.",
    (4, 8): "Net ngang va hai net hai ben cua so 4 tao cam giac hai vong so 8.",
    (7, 9): "Phan dau cong va net doc ben phai cua so 7 giong so 9.",
}
for i in sai:
    plt.figure(figsize=(3, 3))
    plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
    plt.title(f"That {y_te[i]} - Doan {du_doan[i]}")
    plt.axis("off")
    plt.tight_layout()
    duong_dan = thu_muc_anh / f"anh_{i:03d}_that_{y_te[i]}_doan_{du_doan[i]}.png"
    plt.savefig(duong_dan, dpi=180)
    plt.close()
    print(f"  anh {i:3d}: that {y_te[i]}, doan {du_doan[i]}; luu {duong_dan}")
    print("    Nhan xet: " + nhan_xet_anh.get((y_te[i], du_doan[i]),
                                     "Net viet mo hoac bien dang, can xem anh de danh gia."))

print(
    "Cac nhan xet ve hinh dang la quan sat thu cong tren anh 8x8, "
    "khong phai nguyen nhan suy ra truc tiep tu mo hinh."
)
