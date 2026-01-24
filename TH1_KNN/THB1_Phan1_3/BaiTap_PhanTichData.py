import pandas as pd
import numpy as np

# Doc du lieu
df = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\Data\iris_data.csv", sep=',')

print("===== TOAN BO DU LIEU =====")
print(df)
print()

# Hien thi 10 dong dau tien
print("===== 10 DONG DAU TIEN =====")
print(df.head(10))
print()

# Hien thi 5 dong cuoi cung
print("===== 5 DONG CUOI CUNG =====")
print(df.tail())
print()

# Hien thi thong tin dataset
print("===== THONG TIN DATASET =====")
print(df.info())
print()

# Lay cot nhan (label)
y_data = df.loc[:, "nhan"]
print("===== DU LIEU COT NHAN =====")
print(y_data)
print()

# Tap cac gia tri nhan khac nhau
labels = np.unique(y_data)
print("===== CAC GIA TRI NHAN =====")
print(labels)
print()

# Lay du lieu cac thuoc tinh (4 cot dau)
x_data = df.iloc[:, 0:4]
print("===== DU LIEU CAC THUOC TINH =====")
print(x_data)
print()

# Lay du lieu thuoc tinh thu 2
x_data_2 = df.iloc[:, 1]
print("===== DU LIEU THUOC TINH THU 2 =====")
print(x_data_2)
print()

# Dem so dong va so cot
print("===== SO DONG VA SO COT =====")
print("So dong:", df.shape[0])
print("So cot:", df.shape[1])
print()

# Thong ke du lieu
print("===== THONG KE DU LIEU =====")
print(df.describe())
print()
