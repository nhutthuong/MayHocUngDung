import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# DOC DU LIEU

data = pd.read_csv("Data\winequality-white.csv", sep=';')

print("===== THONG TIN DU LIEU =====")
print(data.head())
print("So dong, so cot:", data.shape)
print()

# PHAN TICH DU LIEU


print("===== THONG KE CO BAN =====")
print(data.describe())
print()

print("===== PHAN BO NHAN (quality) =====")
print(data['quality'].value_counts())
print()

# Chuan bi du lieu

X = data.drop("quality", axis=1)
y = data["quality"]

# CHIA DU LIEU: K_fold k=10

kf = KFold(n_splits=10, shuffle=True, random_state=42)

knn = KNeighborsClassifier(
    n_neighbors=9,
    p=2
)
accuracies_knn = []

model = GaussianNB()
accuracies_bayes = []

for fold, (train_idx, test_idx) in enumerate (kf.split(X), 1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    print(f"\nFold {fold}")
    print("So mau tap test:", len(y_test))
    print("Cac gia tri nhan trong tap test:", np.unique(y_test))

    print("Ket qua giai thuat KNN")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    for i in range (12):
        print(f"{i+1}: Nhan thuc te: {y_test.iloc[i]} | Nhan du doan: {y_pred_knn[i]}")

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_12 = accuracy_score(y_test.iloc[:12], y_pred_knn[:12])
    accuracies_knn.append(acc_knn)
    print("Ma tran phan lop:")
    print(confusion_matrix(y_test, y_pred_knn))
    print(f"Do chinh xac cua 12 ph tu dau: {acc_12:.4f}")
    print(f"Do chinh xac cua knn: {acc_knn:.4f}")

    print("Ket qua giai thuat Bayes")

    model.fit(X_train, y_train)
    y_pred_bayes = model.predict(X_test)

    acc_bayes = accuracy_score(y_test, y_pred_bayes)
    accuracies_bayes.append(acc_bayes)

    print(f"Accuracy: {acc_bayes:.4f}")

 
