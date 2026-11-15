"""Bài 5: so rừng và SVC trước/sau khi cân bằng lớp."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def cham(mo_hinh):
    du_doan = mo_hinh.fit(X_tr, y_tr).predict(X_te)
    return (
        precision_score(y_te, du_doan),
        recall_score(y_te, du_doan),
        f1_score(y_te, du_doan),
    )


ket_qua = {}
print("Mo hinh           Cach hoc       Precision  Recall      F1")
for ten, tao_mo_hinh in [
    ("Rung 200 cay", lambda w: RandomForestClassifier(
        n_estimators=200, random_state=0, class_weight=w
    )),
    ("SVM RBF", lambda w: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", class_weight=w)
    )),
]:
    thuong = cham(tao_mo_hinh(None))
    can_bang = cham(tao_mo_hinh("balanced"))
    ket_qua[ten] = sum(abs(a - b) for a, b in zip(thuong, can_bang))
    for cach, diem in [("Binh thuong", thuong), ("Can bang", can_bang)]:
        print(f"{ten:17s} {cach:12s} {diem[0]:9.4f} {diem[1]:7.4f} {diem[2]:7.4f}")
    print(f"  Tong do thay doi tuyet doi cua 3 thuoc do: {ket_qua[ten]:.4f}")

print("Mo hinh thay doi nhieu nhat theo tong do lech 3 thuoc do:", max(ket_qua, key=ket_qua.get))
print(
    "Nhan xet: class_weight='balanced' tang muc phat loi tren lop bo hoc hiem. "
    "So sanh precision/recall cho thay viec bao nhieu hon co the tang recall "
    "nhung lam tang bao nham."
)
