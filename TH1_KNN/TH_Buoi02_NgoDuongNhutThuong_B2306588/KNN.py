# ==========================================
# BAI 01 - KNN PHAN LOAI TAP DU LIEU IRIS
# TH1.2_KNN - May hoc ung dung
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. DOC DU LIEU
# ==========================================

iris = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\Data\iris_data.csv")

print("===== THONG TIN DU LIEU =====")
print(iris.head())
print("Shape:", iris.shape)
print()

# ==========================================
# 2. TACH DAC TRUNG VA NHAN
# ==========================================

X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# Ma hoa nhan
y = y.map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

print("So lop:", len(np.unique(y)))
print("So mau:", len(y))
print()

# ==========================================
# HAM HUAN LUYEN VA DANH GIA KNN
# ==========================================

def run_knn(X_train, X_test, y_train, y_test, p_value, distance_name):
    k_values = [1, 2, 4, 6, 8, 10]
    accuracies = []

    print(f"===== KHOANG CACH {distance_name.upper()} =====")

    for k in k_values:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric='minkowski',
            p=p_value
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"k = {k}, Accuracy = {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()

    return k_values, accuracies


# ==========================================
# 3. CHIA DU LIEU - HOLD OUT 2/3 - 1/3
# ==========================================

print("==========================================")
print("CHIA DU LIEU: HOLD-OUT 2/3 - 1/3")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1/3,
    random_state=42
)

k1, acc_euclid_1 = run_knn(X_train, X_test, y_train, y_test, 2, "euclid")
k1, acc_manhattan_1 = run_knn(X_train, X_test, y_train, y_test, 1, "manhattan")

# Ve bieu do
plt.figure()
plt.plot(k1, acc_euclid_1, marker='o', label='Euclid')
plt.plot(k1, acc_manhattan_1, marker='s', label='Manhattan')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k (Hold-out 2/3 - 1/3)")
plt.legend()
plt.grid()
plt.show()


# ==========================================
# 4. CHIA DU LIEU - 80% - 20%
# ==========================================

print("==========================================")
print("CHIA DU LIEU: 80% - 20%")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

k2, acc_euclid_2 = run_knn(X_train, X_test, y_train, y_test, 2, "euclid")
k2, acc_manhattan_2 = run_knn(X_train, X_test, y_train, y_test, 1, "manhattan")

# Ve bieu do
plt.figure()
plt.plot(k2, acc_euclid_2, marker='o', label='Euclid')
plt.plot(k2, acc_manhattan_2, marker='s', label='Manhattan')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k (80% - 20%)")
plt.legend()
plt.grid()
plt.show()
