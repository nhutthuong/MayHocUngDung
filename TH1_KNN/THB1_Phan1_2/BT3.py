matrx = []
value = 1

for i in range (5):
    row = []
    for j in range(5):
        row.append(value)
        value += 1
    matrx.append(row)

print ("In ma tran:")
for row in matrx:
    print(row)

c = int(input("Nhap vao cot can tinh: "))

if 0 <= c < 5:
    tong_cot = 0
    for i in range (5):
        tong_cot += matrx[i][c]
    print("Tong cac phan tu cua cot la: ", tong_cot)
else:
    print("Chi so khong hop le!!!")
