import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# 1. DOC DU LIEU BOSTON HOUSING
# =====================================================

# File CSV sau khi tai tu Kaggle
data = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\Data\HousingData.csv")
data = data.fillna(data.mean()) #Sua loi NaN trong dataset

print("===== THONG TIN DU LIEU =====")
print(data.head())
print("Shape:", data.shape)
print()

# =====================================================
# 2. TACH DAC TRUNG VA NHAN
# =====================================================

X = data.drop("MEDV", axis=1)   # Dac trung -> "MEDV": ten cot, axis=1 tach theo cot
y = data["MEDV"]                # Gia nha trung binh            axis=0 tach theo hang

# =====================================================
# 3. CHIA DU LIEU 80% - 20%
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 4. KNN HOI QUY - KHONG CHUAN HOA
# =====================================================

k_values = [1, 3, 5, 7, 9]

print("==========================================")
print("KNN HOI QUY - KHONG CHUAN HOA")
print("==========================================")

mae_raw = []
mse_raw = []
rmse_raw = []

for k in k_values:
    knn = KNeighborsRegressor(
        n_neighbors=k,
        metric='minkowski',
        p=2
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    mae_raw.append(mae)
    mse_raw.append(mse)
    rmse_raw.append(rmse)

    print(f"k = {k}")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print()

# =====================================================
# 5. CHUAN HOA DU LIEU (MIN-MAX)
# =====================================================

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Chỉ có fit trên tập train, không fit trên tập test để tránh rò rỉ dữ liệu

# =====================================================
# 6. KNN HOI QUY - SAU KHI CHUAN HOA
# =====================================================

print("==========================================")
print("KNN HOI QUY - SAU KHI CHUAN HOA MIN-MAX")
print("==========================================")

mae_scaled = []
mse_scaled = []
rmse_scaled = []

for k in k_values:
    knn = KNeighborsRegressor(
        n_neighbors=k,
        metric='minkowski',
        p=2
    )

    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    mae_scaled.append(mae)
    mse_scaled.append(mse)
    rmse_scaled.append(rmse)

    print(f"k = {k}")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print()

# =====================================================
# 7. VE BIEU DO SO SANH RMSE
# =====================================================

plt.figure()
plt.plot(k_values, rmse_raw, marker='o', label="Khong chuan hoa")
plt.plot(k_values, rmse_scaled, marker='s', label="Chuan hoa Min-Max")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.title("So sanh RMSE truoc va sau chuan hoa")
plt.legend()
plt.grid()
plt.show()
