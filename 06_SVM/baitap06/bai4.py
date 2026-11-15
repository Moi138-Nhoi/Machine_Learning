"""Bài 4: so lưới thô và lưới mịn quanh C=10, gamma=1."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

luoi = {
    "Tho (muc 5)": {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]},
    "Min": {"C": [3, 10, 30], "gamma": [0.3, 1, 3]},
}
ket_qua = {}
for ten, tham_so in luoi.items():
    tim = GridSearchCV(SVC(kernel="rbf"), tham_so, cv=5)
    tim.fit(X_tr, y_tr)
    kiem_tra = tim.score(X_te, y_te)
    ket_qua[ten] = (tim.best_score_, kiem_tra)
    print(f"{ten}: C = {tim.best_params_['C']}, gamma = {tim.best_params_['gamma']}")
    print(f"  CV 5 phan = {tim.best_score_:.4f}; accuracy kiem tra = {kiem_tra:.4f}")

cv_tho, kt_tho = ket_qua["Tho (muc 5)"]
cv_min, kt_min = ket_qua["Min"]
print(f"CV luoi min {'cao hon' if cv_min > 0.8711 else 'khong cao hon'} 0.8711.")
print(f"Accuracy kiem tra {'thay doi' if kt_min != kt_tho else 'khong doi'} so voi luoi tho.")
print(
    "Nhan xet: luoi min co the tim tham so cho diem CV nhich len, "
    "nhung ton them lan fit va khong bao dam accuracy kiem tra tot hon."
)
