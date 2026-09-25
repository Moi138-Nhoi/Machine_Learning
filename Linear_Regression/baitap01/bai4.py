
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("gia_nha.csv")

# Hai biến đầu vào
X = df[["dien_tich", "so_phong"]]

# Biến cần dự đoán
y = df["gia"]

# Chia dữ liệu thành tập học và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tạo mô hình
mo_hinh = LinearRegression()

# Huấn luyện
mo_hinh.fit(X_train, y_train)

# Dự đoán
y_pred = mo_hinh.predict(X_test)

# Tính R2
r2 = r2_score(y_test, y_pred)

print(f"R2 tren tap kiem tra = {r2:.4f}")

print()
print("R2 cua mo hinh mot bien = 0.9622")

if r2 > 0.9622:
    print("Nhan xet: Them bien so_phong lam R2 tang so voi mo hinh chi dung dien_tich.")
elif r2 < 0.9622:
    print("Nhan xet: Them bien so_phong lam R2 giam so voi mo hinh chi dung dien_tich.")
else:
    print("Nhan xet: R2 khong thay doi.")