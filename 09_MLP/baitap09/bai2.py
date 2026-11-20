"""Bài 2: kiểm chứng mạng hai nơ-ron ẩn ReLU tính XOR."""

import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
h1 = np.maximum(0, X[:, 0] - X[:, 1])
h2 = np.maximum(0, X[:, 1] - X[:, 0])
y = h1 + h2
nhan_xor = (X[:, 0] != X[:, 1]).astype(float)

print("x1 x2  h1=max(0,x1-x2)  h2=max(0,x2-x1)  y=h1+h2  XOR")
for x, a, b, du_doan, that in zip(X, h1, h2, y, nhan_xor):
    print(f" {x[0]:.0f}  {x[1]:.0f} {a:17.0f} {b:17.0f} {du_doan:8.0f} {that:4.0f}")
print("Khop ca bon truong hop:", np.array_equal(y, nhan_xor))
print("Giai thich: neu x1=x2, ca hai hieu bang 0 nen y=0.")
print("Neu x1 khac x2, dung mot hieu bang +1 va hieu kia bang -1;")
print("ReLU giu +1, cat -1 ve 0, nen h1+h2=1.")
