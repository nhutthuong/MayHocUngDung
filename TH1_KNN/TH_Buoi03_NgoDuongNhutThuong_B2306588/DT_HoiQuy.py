import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "D:\Lap trinh\MayHocUngDung\TH1_KNN\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\housing.csv",
    header=None,
    sep=r"\s+"
)

#Hien thi 5 dong dau tien
print(df.head())

#Tach dac trung va nhan
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Chia du lieu thanh tap huan luyen va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)