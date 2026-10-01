"""Bài 2: dùng mô hình giả định để tính xác suất qua môn."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


w, b = 0.4, -5.0
for gio_on in [5, 10, 12.5, 15, 20, 25]:
    p = sigmoid(w * gio_on + b)
    print(f"{gio_on:>4} giờ: P(qua môn)={p:.3f}, dự đoán={(p >= 0.5)}")
