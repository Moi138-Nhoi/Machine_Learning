"""Bước 1: đọc dữ liệu sinh viên và xem qua một lượt."""
import pandas as pd

df = pd.read_csv("sinh_vien.csv")

print("Kích thước bảng (số dòng, số cột):", df.shape)
print("Năm dòng đầu tiên:")
print(df.head())
print("\nThông tin các cột:")
df.info()
print("\nThống kê mô tả:")
print(df.describe())
print("\nSố sinh viên theo kết quả:")
print(df["qua_mon"].value_counts().sort_index())
print("\nTỷ lệ qua môn:", round(df["qua_mon"].mean(), 4))
print("\nSố giờ ôn trung bình theo kết quả:")
print(df.groupby("qua_mon")["gio_on"].mean().round(2))
