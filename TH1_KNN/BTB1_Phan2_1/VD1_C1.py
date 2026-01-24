from pandas import read_csv
import numpy as np

iris = read_csv('D:\Lap trinh\MayHocUngDung\TH1_KNN\Data\iris_data.csv', delimiter=',', index_col=0)

iris_x = iris.iloc[:, :-1]
iris_y = iris.iloc[:, -1]

iris_y = iris_y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

print("Number of class: ", len(np.unique(iris_y)))
print("Number of data: ", len(iris_y))
print()

for i in range(3):
    print(f"Sample data from class {i}:")
    print(iris_x[iris_y == i].head())
    print()