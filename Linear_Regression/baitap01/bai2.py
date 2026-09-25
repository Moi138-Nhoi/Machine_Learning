
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gia_nha.csv")

# Vẽ biểu đồ phân tán
plt.scatter(df["so_phong"], df["gia"])

# Đặt tên cho trục
plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")

# Đặt tiêu đề
plt.title("Gia nha theo so phong")

# Lưu hình
plt.savefig("bai2.png", dpi=150)

# Hiển thị hình
plt.show()

print("Nhan xet: So phong co xu huong tang thi gia nha cung co xu huong tang, tuy nhien cac diem khong nam hoan toan tren mot duong thang.")