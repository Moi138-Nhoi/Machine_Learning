"""Học cây năm lá, đọc luật và dự đoán cho hai khách mới."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


df = pd.read_csv(Path(__file__).parent / "data/khach_vay.csv")
print("Kich thuoc bang:", df.shape)
print(df.head())
print("\nTy le khach tra dung han:", round(df["tra_dung_han"].mean(), 2))

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cay = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
print(f"\nDo sau {cay.get_depth()}, so la {cay.get_n_leaves()}")
print(export_text(cay, feature_names=list(X.columns), show_weights=True, decimals=3))
print(f"Accuracy tap hoc      : {accuracy_score(y_tr, cay.predict(X_tr)):.4f}")
print(f"Accuracy tap kiem tra : {accuracy_score(y_te, cay.predict(X_te)):.4f}\n")

moi = pd.DataFrame({
    "thu_nhap": [20.0, 10.0],
    "ty_le_no": [0.4, 0.3],
    "co_viec_on_dinh": [1, 0],
    "so_nguoi_phu_thuoc": [2, 1],
})
du_doan = cay.predict(moi)
xac_suat = cay.predict_proba(moi)[:, 1]
for i in range(len(moi)):
    ket = "dung han" if du_doan[i] == 1 else "tre han"
    print(f"Khach {i + 1}: {ket}, xac suat dung han {xac_suat[i]:.2f}")
