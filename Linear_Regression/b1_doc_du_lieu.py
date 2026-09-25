# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 1: đọc tệp dữ liệu và nhìn qua một lượt.

 Chú thích trong tệp viết tiếng Việt có dấu, nhưng phần in ra màn hình cố ý
 viết không dấu. Cửa sổ lệnh Windows thường không hiện được chữ có dấu.
 """

import pandas as pd

# Đọc tệp CSV thành một bảng. Tên df là viết tắt quen dùng của data frame.
df = pd.read_csv("gia_nha.csv") #đọc tệp csv vào biến df

print("Kich thuoc bang (so dong, so cot):", df.shape) #df.shape: trả về 1 cặp số là kích thước của bảng
print()

print("Nam dong dau tien:")
print(df.head()) #trả về 5 dòng đầu tiên của bảng
print()
print("Ten cac cot:", list(df.columns)) #liệt kê các tên cột trong bảng
print()

print("Thong ke nhanh cot dien_tich va cot gia:")
print(df[["dien_tich", "gia"]].describe().round(2))
print()

print("So o bi thieu trong tung cot:")
print(df.isna().sum())

