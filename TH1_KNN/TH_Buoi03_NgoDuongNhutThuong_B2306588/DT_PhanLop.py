import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

#Doc du lieu tu dataset
df = pd.read_excel("D:\Lap trinh\MayHocUngDung\TH1_KNN\TH_Buoi03_NgoDuongNhutThuong_B2306588\Data\Play.xlsx", engine='openpyxl')

#Hien thi thong tin ve dataframe
df.info()

#In ra 5 dong dau tien trong tap du lieu
print(df.head())

#Tach dac trung va nhan
X = df[["Weather", "Temp"]]
y = df["Play"]

y=y.str.strip()

#Chia du lieu
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("So dong train:", len(X_train))
print("So dong test:", len(X_test))

#Duyet qua tung cot trong tap du lieu huan luyen
for col in X_train.columns:

    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Ve cay quyet dinh
plt.figure(figsize=(8,5)) #Kich thuoc hinh

plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=True,
    filled=True,
    rounded=True
)

plt.show() #Hien thi cay quyet dinh

#Danh gia do chinh xac
accuracy = accuracy_score(y_test, y_pred)
print(f"Do chinh xac: {accuracy:.2f}")

#Ma tran du doan
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=model.classes_,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test,y_pred))