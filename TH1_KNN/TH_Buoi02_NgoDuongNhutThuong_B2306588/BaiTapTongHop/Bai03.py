import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================
# a. DOC DU LIEU
# =====================================================

data = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\Data\winequality-white.csv", sep=';')

print("===== THONG TIN DU LIEU =====")
print(data.head())
print("So dong, so cot:", data.shape)
print()

# =====================================================
# b. PHAN TICH DU LIEU
# =====================================================

print("===== THONG KE CO BAN =====")
print(data.describe())
print()

print("===== PHAN BO NHAN (quality) =====")
print(data['quality'].value_counts())
print()

# =====================================================
# Chuyen ve bai toan PHAN LOAI
# quality >= 6 -> ruou tot (1)
# quality < 6  -> ruou khong tot (0)
# =====================================================

data['quality_label'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# =====================================================
# c. CHIA DU LIEU: 8 PHAN TRAIN - 2 PHAN TEST (80% - 20%)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("So mau tap test:", len(y_test))
print("Cac gia tri nhan trong tap test:", np.unique(y_test))
print()

# =====================================================
# d. KNN PHAN LOAI - k = 7 (KHONG CHUAN HOA)
# =====================================================

knn = KNeighborsClassifier(
    n_neighbors=7,
    metric='minkowski',
    p=2
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("==========================================")
print("KNN - KHONG CHUAN HOA")
print("==========================================")

print("Do chinh xac tong the:", accuracy_score(y_test, y_pred))
print()

# In 8 phan tu dau tien tap test
print("8 phan tu dau tien tap test:")
for i in range(8):
    print(f"Thuc te: {y_test.iloc[i]}, Du doan: {y_pred[i]}")

print("Do chinh xac 8 phan tu dau:",
      accuracy_score(y_test.iloc[:8], y_pred[:8]))
print()

print("Ma tran phan lop:")
print(confusion_matrix(y_test, y_pred))
print()

print("Bao cao chi tiet:")
print(classification_report(y_test, y_pred))

# =====================================================
# CHUAN HOA DU LIEU MIN-MAX
# =====================================================

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# KNN PHAN LOAI - k = 7 (SAU CHUAN HOA)
# =====================================================

knn_scaled = KNeighborsClassifier(
    n_neighbors=7,
    metric='minkowski',
    p=2
)

knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("==========================================")
print("KNN - SAU CHUAN HOA MIN-MAX")
print("==========================================")

print("Do chinh xac tong the:", accuracy_score(y_test, y_pred_scaled))
print()

print("Ma tran phan lop:")
print(confusion_matrix(y_test, y_pred_scaled))
print()

print("Bao cao chi tiet:")
print(classification_report(y_test, y_pred_scaled))
