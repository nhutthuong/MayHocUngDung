import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#DOC DU LIEU

data = pd.read_csv("Data\winequality-red.csv", sep=';')

print("===== THONG TIN DU LIEU =====")
print(data.head())
print("So dong, so cot:", data.shape)
print()

# PHAN TICH DU LIEU 

print("===== PHAN BO NHAN (quality) =====")
print(data['quality'].value_counts()) #Đếm số lần xuất hiện của các giá trị
print(data['quality'])
print()

# Chuan bi du lieu

X = data.drop("quality", axis=1)
y = data["quality"]

# CHIA DU LIEU: 4 PHAN TRAIN - 1 PHAN TEST (80% - 20%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y #giúp đảm bảo phân bố nhãn trong tập train và test tương tự như dữ liệu gốc, 
                #đặc biệt quan trọng trong bài toán phân loại.
)

print("So mau tap test:", len(y_test))
print("Cac gia tri nhan trong tap test:", y_test.value_counts())
print("")
print()

# CHUAN HOA DU LIEU MIN-MAX

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN PHAN LOAI - k = 5

knn_scaled = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)

knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("Ket qua giai thuat KNN")

print("Do chinh xac tong the:", accuracy_score(y_test, y_pred_scaled))
print()

print("Ma tran phan lop:")
print(confusion_matrix(y_test, y_pred_scaled))
print()

print("Bao cao chi tiet:")
print(classification_report(y_test, y_pred_scaled,zero_division=0))


# Giai thuat Bayes

model = GaussianNB()

model.fit(X_train, y_train)
y_pred_Bayes = model.predict(X_test)

print("Ket qua giai thuat Bayes")

print("Do chinh xac tong the:", accuracy_score(y_test, y_pred_Bayes))
print()

print("Do chinh xac tung phan lop")
print(classification_report(y_test, y_pred_Bayes))
