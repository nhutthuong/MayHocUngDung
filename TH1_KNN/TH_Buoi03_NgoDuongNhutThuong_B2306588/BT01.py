import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Doc tap du lieu
df = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\drug200.csv", sep=',')

#Mo ta tap DL
print ("=========MO TA TAP DU LIEU=========")
print("So phan tu:", df.shape[0])
print ("So thuoc tinh:", df.shape[1]-1)
print("Nhan (target):", df.columns[-1])
print("Cac gia tri nhan:")
print(df['Drug'].value_counts())
print("Thong tin 5 dong dau du lieu:")
print(df.head())

#Tien xu ly

le = LabelEncoder()

# Encode các thuộc tính dạng chữ
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Drug", axis=1)
y = df["Drug"]

# THU NGHIEM CAC TRUONG HOP

train_test_ratios = [(0.6, 0.4), (0.8, 0.2)]
min_samples_list = [5, 10, 15, 20]

best_accuracy = 0
best_config = None

print("=========BAT DAU THU NGHIEM CAC TRUONG HOP=========")

for train_ratio, test_ratio in train_test_ratios:

    print(f"\n==== CHIA {int(train_ratio*100)}% TRAIN - {int(test_ratio*100)}% TEST ====")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=42,
        stratify=y
    )

    for min_samples in min_samples_list:

        model = DecisionTreeClassifier(
            criterion='gini',              # theo yêu cầu đề
            min_samples_leaf=min_samples,  # số mẫu tối thiểu tại lá
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"min_samples_leaf = {min_samples} --> Accuracy = {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_config = (
                f"{int(train_ratio*100)}% train - "
                f"min_samples_leaf={min_samples}"
            )

#KET LUAN MO HINH TOT NHAT
print("=========MO HINH TOT NHAT=========")
print("Cau hinh tot nhat:", best_config)
print("Do chinh xac cao nhat:", best_accuracy)