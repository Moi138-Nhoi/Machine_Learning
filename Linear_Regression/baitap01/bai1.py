
import pandas as pd

df = pd.read_csv("gia_nha.csv")

# Lọc các căn có diện tích lớn hơn 100 m2
nhom_lon = df[df["dien_tich"] > 100]

# Đếm số căn
so_can = len(nhom_lon)

# Tính giá trung bình
gia_trung_binh = nhom_lon["gia"].mean()

print("So can ho co dien tich lon hon 100 m2:", so_can)
print(f"Gia trung binh cua nhom: {gia_trung_binh:.4f} ty dong")