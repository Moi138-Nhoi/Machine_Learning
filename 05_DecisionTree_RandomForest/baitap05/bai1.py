"""Bài 1: vẽ cây năm lá và cây sâu hai, rồi đọc hai lá bên trái."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Lưu hình được cả khi chạy không có cửa sổ đồ họa.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


thu_muc = Path(__file__).resolve().parent
df = pd.read_csv(thu_muc.parent / "data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, _, y_tr, _ = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def ve_cay(cay, ten_tep, tieu_de):
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        cay, feature_names=list(X.columns), class_names=["Tre han", "Dung han"],
        filled=True, rounded=True, fontsize=9, ax=ax
    )
    ax.set_title(tieu_de)
    fig.tight_layout()
    duong_dan = thu_muc / ten_tep
    fig.savefig(duong_dan, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Da luu:", duong_dan)


cay_5 = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
ve_cay(cay_5, "cay_nam_la.png", "Cay quyet dinh nam la")

cay_2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
ve_cay(cay_2, "cay_sau_2.png", "Cay quyet dinh sau hai")

# Đọc hai lá của nhánh trái để giải thích vì sao cây vẫn chia.
t = cay_2.tree_
nut_trai = t.children_left[0]
hai_la = [t.children_left[nut_trai], t.children_right[nut_trai]]
lop = [int(cay_2.classes_[t.value[i, 0].argmax()]) for i in hai_la]
print("Ket luan hai la ben trai:", lop)
print("Hai la cung ket luan:", "co" if lop[0] == lop[1] else "khong")
print(f"Gini nut truoc chia: {t.impurity[nut_trai]:.4f}")
for i, nut in enumerate(hai_la, 1):
    print(f" La {i}: {t.n_node_samples[nut]} khach, Gini {t.impurity[nut]:.4f}")
gini_sau = sum(t.n_node_samples[nut] * t.impurity[nut] for nut in hai_la) / t.n_node_samples[nut_trai]
print(f"Gini sau chia co trong so: {gini_sau:.4f}; giam {t.impurity[nut_trai] - gini_sau:.4f}")
print("Nhan xet: Cay chia neu Gini co trong so giam, du hai la co the "
      "cung lop da so; ti le hai lop trong hai la van khac nhau.")
