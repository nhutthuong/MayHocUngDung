m = int(input("Nhap vao so hang m = "))
n = int(input("Nhap vao so cot n = "))

matrix = []

for i in range (m):
    row = []
    for j in range (n):
        value = int(input())
        row.append(value)
    matrix.append(row)

h = int(input("Nhap vao hang can tinh: "))
if 0 <= h < m:
    tong = sum(matrix[h])
    print("Tong cac phan tu cua hang la: ", tong)
else:
    print("Chi so hang khong hop le!!!")


