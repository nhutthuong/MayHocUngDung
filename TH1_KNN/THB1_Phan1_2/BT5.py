# Nhập kích thước ma trận A
M = int(input("Nhập số hàng của ma trận A: "))
N = int(input("Nhập số cột của ma trận A: "))

# Nhập ma trận A (M x N)
print("Nhập ma trận A:")
A = []
for i in range(M):
    row = []
    for j in range(N):
        row.append(float(input(f"A[{i}][{j}]: ")))
    A.append(row)

print ("In ma tran A:")
for row in A:
    print(row)

c1 = int(input("Nhập cột thứ nhất cần đổi của ma trận A: "))
c2 = int(input("Nhập cột thứ hai cần đổi của ma trận A: "))

for i in range (M):
    temp = A[i][c1]
    A[i][c1] = A[i][c2]
    A[i][c2] = temp

print ("In ma tran A:")
for row in A:
    print(row)