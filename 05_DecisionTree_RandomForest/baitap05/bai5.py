"""Bài 5: so sánh độ quan trọng Gini với permutation importance."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Dùng cùng bộ sinh số: 450 số đầu cho tập học, 150 số tiếp cho tập kiểm tra.
rng = np.random.default_rng(0)
X_tr_nn = X_tr.copy()
X_te_nn = X_te.copy()
X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr_nn))
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te_nn))
rung = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr_nn, y_tr)

gini_qt = pd.Series(rung.feature_importances_, index=X_tr_nn.columns)
hoan_vi = permutation_importance(
    rung, X_te_nn, y_te, scoring="accuracy", n_repeats=20, random_state=0
)
hoan_vi_qt = pd.Series(hoan_vi.importances_mean, index=X_tr_nn.columns)

print(f"Accuracy goc tren tap kiem tra: {accuracy_score(y_te, rung.predict(X_te_nn)):.4f}")
print("\nDo quan trong theo Gini (feature_importances_)")
print(gini_qt.sort_values(ascending=False).round(4).to_string())
print("\nDo giam accuracy khi xao tron (trung binh 20 lan)")
print(hoan_vi_qt.sort_values(ascending=False).round(4).to_string())

hang_gini = int(gini_qt.rank(ascending=False, method="min")["so_ngau_nhien"])
hang_hoan_vi = int(hoan_vi_qt.rank(ascending=False, method="min")["so_ngau_nhien"])
print(f"\nThu hang cot ngau nhien: Gini {hang_gini}/5, xao tron {hang_hoan_vi}/5")
print("Giai thich: Gini cong diem qua moi phep chia tren tap hoc, nen cot lien tuc "
      "nhieu nguong de duoc diem ca khi chi khop nhieu. Xao tron tren tap kiem tra "
      "do muc giam accuracy ngoai mau; cot nhieu khong giup du doan se co diem gan 0 "
      "(co the am do dao dong mau). Do quan trong khong chung minh quan he nhan qua.")
