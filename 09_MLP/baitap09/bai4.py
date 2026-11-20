"""Bài 4: so tốc độ học và vẽ loss_curve_ của mạng chữ số."""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Learning rate   So vong   Loss cuoi   Accuracy kiem tra   Hoi tu")
plt.figure(figsize=(8, 5))
ket_qua = {}
for toc_do in [0.0001, 0.001, 0.01, 0.1]:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=toc_do,
                      max_iter=1000, random_state=0),
    )
    with warnings.catch_warnings(record=True) as canh_bao:
        warnings.simplefilter("always", ConvergenceWarning)
        mang.fit(X_tr, y_tr)
    mlp = mang[-1]
    hoi_tu = not any(issubclass(w.category, ConvergenceWarning) for w in canh_bao)
    acc = accuracy_score(y_te, mang.predict(X_te))
    ket_qua[toc_do] = (mlp.n_iter_, mlp.loss_curve_[-1], acc)
    print(f"{toc_do:13g} {mlp.n_iter_:9d} {mlp.loss_curve_[-1]:11.6f}"
          f" {acc:19.4f} {str(hoi_tu):>8s}")
    plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_,
             label=f"learning_rate_init={toc_do:g}")

plt.yscale("log")
plt.xlabel("Vong hoc")
plt.ylabel("Loss (thang log)")
plt.title("Loss cua MLP tren bo chu so")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "toc_do_hoc.png", dpi=180)
plt.close()
print("Da luu toc_do_hoc.png")
print(f"Nhan xet: 0.0001 can {ket_qua[0.0001][0]} vong va dat accuracy "
      f"{ket_qua[0.0001][2]:.4f}; toc do nho hoc lau hon.")
print(f"Toc do 0.1 dung sau {ket_qua[0.1][0]} vong nhung loss cuoi "
      f"{ket_qua[0.1][1]:.4f}, accuracy {ket_qua[0.1][2]:.4f}; "
      "duong loss dao dong ro, cho thay buoc hoc qua lon.")
