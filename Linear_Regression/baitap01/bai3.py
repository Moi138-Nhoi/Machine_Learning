
import pandas as pd

df = pd.read_csv("gia_nha.csv")

# Biến đầu vào là tuổi nhà
x = df["tuoi_nha"].to_numpy()

# Biến cần dự đoán là giá
y = df["gia"].to_numpy()

# Tính giá trị trung bình
x_tb = x.mean()
y_tb = y.mean()

# Công thức bình phương tối thiểu
tu_so = ((x - x_tb) * (y - y_tb)).sum()
mau_so = ((x - x_tb) ** 2).sum()

w = tu_so / mau_so
b = y_tb - w * x_tb

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")

print()

if w < 0:
    print("Nhan xet: w mang dau am.")
    print("Dieu nay cho thay khi tuoi nha tang them 1 nam, gia du doan co xu huong giam.")
else:
    print("Nhan xet: w khong mang dau am.")