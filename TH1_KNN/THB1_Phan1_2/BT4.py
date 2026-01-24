matrx = []
value = 1

for i in range (4):
    row = []
    for j in range(4):
        row.append(value)
        value += 2
    matrx.append(row)

print ("In ma tran:")
for row in matrx:
    print(row)

tong_hang2 = sum(matrx[1])
print("Tong cac phan tu cua cot la: ", tong_hang2)
