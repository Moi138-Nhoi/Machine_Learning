"""Bài 5: dò lưới kiến trúc và alpha của MLP trên bộ chữ số."""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
pipeline = make_pipeline(
    StandardScaler(), MLPClassifier(max_iter=1000, random_state=0)
)
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1.0],
}
tim = GridSearchCV(pipeline, luoi, cv=5, scoring="accuracy", n_jobs=1)
tim.fit(X_tr, y_tr)

bang = pd.DataFrame({
    "lop_an": [str(p["mlpclassifier__hidden_layer_sizes"]) for p in tim.cv_results_["params"]],
    "alpha": [p["mlpclassifier__alpha"] for p in tim.cv_results_["params"]],
    "cv_accuracy": tim.cv_results_["mean_test_score"],
})
print("Accuracy kiem dinh cheo 5 phan (chi tren tap hoc):")
print(bang.pivot(index="lop_an", columns="alpha", values="cv_accuracy").round(4).to_string())
print("Tham so tot nhat:", tim.best_params_)
print(f"CV tot nhat: {tim.best_score_:.4f}")
acc_mlp = tim.score(X_te, y_te)
acc_svm = SVC().fit(X_tr, y_tr).score(X_te, y_te)
print(f"MLP tren tap kiem tra: {acc_mlp:.4f}")
print(f"SVM RBF tren cung tap kiem tra: {acc_svm:.4f}")
print("MLP vuot SVM:", acc_mlp > acc_svm)
print("Nhan xet: chi chon tham so bang CV tren tap hoc; tap kiem tra")
print("duoc dung mot lan sau khi GridSearchCV hoc lai mo hinh tot nhat.") 