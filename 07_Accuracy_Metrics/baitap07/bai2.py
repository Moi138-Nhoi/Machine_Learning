"""Bài 2: báo cáo phân loại SVM cho 10 chữ số viết tay."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
du_doan = SVC(kernel="rbf").fit(X_tr, y_tr).predict(X_te)
bao_cao = classification_report(y_te, du_doan, digits=4, output_dict=True)
print(classification_report(y_te, du_doan, digits=4))

so_recall_thap = min(range(10), key=lambda so: bao_cao[str(so)]["recall"])
so_precision_thap = min(range(10), key=lambda so: bao_cao[str(so)]["precision"])
print(f"Recall thap nhat: so {so_recall_thap}, {bao_cao[str(so_recall_thap)]['recall']:.4f}")
print(f"Precision thap nhat: so {so_precision_thap}, {bao_cao[str(so_precision_thap)]['precision']:.4f}")
print("So anh kiem tra moi chu so:", np.bincount(y_te).tolist())
print(
    "Nhan xet: moi chu so co so anh gan bang nhau, nen macro avg "
    "(trung binh deu) va weighted avg (theo so anh) gan nhu trung nhau."
)
