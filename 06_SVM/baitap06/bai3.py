"""Bài 3: dò C cho kernel đa thức trên dữ liệu hai vầng trăng."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = []
print("Bac  C tot nhat  CV 5 phan  Accuracy kiem tra")
for bac in [2, 3, 4]:
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=bac))
    tim = GridSearchCV(pipeline, {"svc__C": [0.1, 1, 10, 100]}, cv=5)
    tim.fit(X_tr, y_tr)
    accuracy = tim.score(X_te, y_te)
    ket_qua.append((bac, tim.best_score_, accuracy))
    print(f"{bac:3d}  {tim.best_params_['svc__C']:10g}  {tim.best_score_:10.4f}  {accuracy:18.4f}")

# Cùng quy trình mục 5: RBF không chuẩn hoá, dò C và gamma trên tập học.
rbf = GridSearchCV(
    SVC(kernel="rbf"), {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}, cv=5
)
rbf.fit(X_tr, y_tr)
rbf_accuracy = rbf.score(X_te, y_te)
bac_tot, cv_tot, accuracy_tot = max(ket_qua, key=lambda hang: hang[1])
print(f"Bac tot nhat theo CV: {bac_tot} (CV = {cv_tot:.4f}).")
print(f"RBF muc 5: CV = {rbf.best_score_:.4f}; accuracy kiem tra = {rbf_accuracy:.4f}.")
print(
    "Nhan xet: kernel da thuc "
    + ("vuot" if accuracy_tot > rbf_accuracy else "khong vuot")
    + " RBF tren tap kiem tra; so sanh dua tren cung phep chia du lieu."
)
