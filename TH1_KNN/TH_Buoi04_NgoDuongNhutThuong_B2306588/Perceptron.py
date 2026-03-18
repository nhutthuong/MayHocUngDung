import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, r2_score
import math

df = pd.read_csv("D:\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\data_per.csv")

#Hien thi 5 dong dau tien trong DL
print(df.head())

#Hien thi dong co index = 3
print("_" * 50)
print(df.loc[3])

#Tach dac trung va nhan
X = df.iloc[:, :-1]
y=df.iloc[:,-1]

#Chia du lieu
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=2,
    random_state=42
)

#Khoi tao mo hinh
net = Perceptron()

net.fit(X_train, y_train)
y_pred = net.predict(X_test)

print(f"Cac thuoc tinh: {X.columns.values}")
print(f"He so cua cac thuoc tinh: {net.coef_}")
print(f"Trong so w0 (intercept): {net.intercept_}")
print(f"So lan lap: {net.n_iter_}")

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)