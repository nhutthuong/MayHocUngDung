import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Data\Student_Performance.csv")

print("===== THONG TIN TAP DU LIEU =====")
print("So phan tu:", data.shape[0])
print("So thuoc tinh:", data.shape[1]-1)
print("Nhan (target): Performance Index")
print(data.head())

#Tien xu ly

le = LabelEncoder()

# Encode các thuộc tính dạng chữ
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# TACH DU LIEU

X = data.drop("Performance Index", axis=1)
y = data["Performance Index"]

# K-FOLD (K = 50, shuffle=True)

kf = KFold(n_splits=50, shuffle=True, random_state=42)

fold = 1

mae_list = []
mse_list = []
rmse_list = []
r2_list = []

for train_index, test_index in kf.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    if fold == 1:
        print("\n===== THONG TIN TAP TRAIN/TEST =====")
        print("So phan tu train:", len(X_train))
        print("So phan tu test:", len(X_test))

    # XAY DUNG MO HINH HOI QUY TUYEN TINH

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # TINH CAC CHI SO SAI SO

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"\nFold {fold}")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    fold += 1

# SAI SO TRUNG BINH

print("\n===================================")
print("SAI SO TRUNG BINH 50 FOLD")
print("===================================")

print("MAE trung binh :", np.mean(mae_list))
print("MSE trung binh :", np.mean(mse_list))
print("RMSE trung binh:", np.mean(rmse_list))
print("R2 trung binh  :", np.mean(r2_list))