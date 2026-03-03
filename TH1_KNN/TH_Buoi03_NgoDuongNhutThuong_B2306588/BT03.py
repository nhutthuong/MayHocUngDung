import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(
    "D:\Lap trinh\MayHocUngDung\TH1_KNN\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\housing.csv",
    header=None,
    sep=r"\s+"
)

#Mo ta tap DL
print ("=========MO TA TAP DU LIEU=========")
print("So phan tu:", df.shape[0])
print ("So thuoc tinh:", df.shape[1]-1)
print("Nhan (target):", df.columns[-1])
print("Cac gia tri nhan:")
print(df.iloc[:, -1].value_counts())
print("Thong tin 5 dong dau du lieu:")
print(df.head())

#Tach dac trung va nhan
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

randomState = [1,3,5,7,9,11,13,15,17,19]

MSE_tb = 0
R2_tb = 0
for rdState in randomState:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=rdState
    )
    #Chuan hoa DL
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Huan luyen mo hinh
    model = DecisionTreeRegressor(
        criterion='absolute_error',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nRandom State = {rdState}")
    print (f"MSE: {mse:.2f}")
    print (f"R2: {r2:.2f}")

    MSE_tb += mse
    R2_tb += r2

MSE_tb /= 10
R2_tb /= 10

print (f"MSE trung binh: {MSE_tb:.2f}")
print (f"R2 trung binh: {R2_tb:.2f}")