import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

df = pd.read_csv("D:\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\housing_RT.csv", index_col=0)

#Hien thi 5 dong dau tien trong DL
print(df.head())

#Hien thi dong co index = 3
print("_" * 50)
print(df.loc[3])

#Tach dac trung va nhan
X = df.iloc[:, 1:5]
y=df.iloc[:,0]

#Chia du lieu
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=20,
    random_state=42
)

#Khoi tao mo hinh
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(f"Cac thuoc tinh: {X.columns.values}")
print(f"He so cua cac thuoc tinh: {model.coef_}")
print(f"Bias (intercept): {model.intercept_}")

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)