import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#Doc tap du lieu
df = pd.read_csv("D:\Lap trinh\MayHocUngDung\TH1_KNN\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\winequality-white.csv", sep=';')

#Mo ta tap DL
print ("=========MO TA TAP DU LIEU=========")
print("So phan tu:", df.shape[0])
print ("So thuoc tinh:", df.shape[1]-1)
print("Nhan (target):", df.columns[-1])
print("Cac gia tri nhan:")
print(df['quality'].value_counts())
print("Thong tin 5 dong dau du lieu:")
print(df.head())

X = df.drop("quality", axis=1)
y = df["quality"]

# CHIA DU LIEU: K_fold k=50

kf = KFold(n_splits=50, shuffle=True, random_state=42)


model = DecisionTreeClassifier(
    criterion='entropy',              # theo yêu cầu đề
    min_samples_leaf=400,  # số mẫu tối thiểu tại lá
    random_state=42
)
accuracy = 0

for fold, (train_idx, test_idx) in enumerate (kf.split(X), 1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    print(f"\nFold {fold}")
    print("So mau tap test:", len(y_test))
    print("So mau tap huan luyen:", len(y_train))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    #Do chinh xac tung phan lop
    print("Do chinh xac tung phan lop:")
    print(classification_report(y_test,y_pred,zero_division=0))
    print(f"Do chinh xac tong the: {acc:.2f}")
    accuracy = accuracy + acc

accuracy = accuracy/50
print(f"\nDo chinh xac trung binh cua 50 lap: {accuracy:.2f}")