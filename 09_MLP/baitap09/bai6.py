"""Bài 6: so hồi quy tuyến tính với MLP trên giá nhà."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

duong_thang = LinearRegression().fit(X_tr, y_tr)
mang = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
).fit(X_tr, y_tr)

mae_duong = mean_absolute_error(y_te, duong_thang.predict(X_te))
mae_mang = mean_absolute_error(y_te, mang.predict(X_te))
can_moi = pd.DataFrame({"dien_tich": [250.0]})
gia_duong = duong_thang.predict(can_moi)[0]
gia_mang = mang.predict(can_moi)[0]
print(f"So can hoc/kiem tra: {len(X_tr)}/{len(X_te)}")
print(f"Dien tich lon nhat trong tap hoc: {X_tr['dien_tich'].max():.1f} m2")
print(f"MAE duong thang: {mae_duong:.4f} ty dong")
print(f"MAE mang MLP   : {mae_mang:.4f} ty dong")
print(f"Du doan 250 m2 - duong thang: {gia_duong:.4f} ty dong")
print(f"Du doan 250 m2 - MLP        : {gia_mang:.4f} ty dong")
print(f"Chenh lech hai du doan: {abs(gia_mang-gia_duong):.4f} ty dong")
print(f"MLP co MAE {mae_mang/mae_duong:.2f} lan MAE duong thang tren tap kiem tra.")
print("Nhan xet: 250 m2 nam ngoai khoang dien tich da hoc, nen MAE")
print("tren tap kiem tra khong chung minh do chinh xac tai 250 m2.")
print("Du doan MLP tai 250 m2 lech xa duong thang, nen tren vi du nay")
print("MLP khong ngoai suy on dinh nhu mo hinh tuyen tinh.")
