"""Bài 1: cài đặt hàm sigmoid và xem một số giá trị."""
import numpy as np


def sigmoid(z):
    """Tính 1 / (1 + exp(-z)) với đầu vào vô hướng hoặc mảng."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)  # tránh tràn số khi tính exp
    return 1.0 / (1.0 + np.exp(-z))


for z in [-6, -4, -2, -1, 0, 1, 2, 4, 6]:
    print(f"sigmoid({z:>2}) = {sigmoid(z):.4f}")
