# Nhập kích thước ma trận A
M = int(input("Nhập số hàng của ma trận A: "))
N = int(input("Nhập số cột của ma trận A: "))

# Nhập kích thước ma trận B
P = int(input("Nhập số cột của ma trận B: "))

# Nhập ma trận A (M x N)
print("Nhập ma trận A:")
A = []
for i in range(M):
    row = []
    for j in range(N):
        row.append(float(input(f"A[{i}][{j}]: ")))
    A.append(row)

# Nhập ma trận B (N x P)
print("Nhập ma trận B:")
B = []
for i in range(N):
    row = []
    for j in range(P):
        row.append(float(input(f"B[{i}][{j}]: ")))
    B.append(row)

# Khởi tạo ma trận kết quả C (M x P)
C = [[0 for _ in range(P)] for _ in range(M)]

# Nhân ma trận
for i in range(M):
    for j in range(P):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]

# In kết quả
print("Ma trận tích C = A x B:")
for row in C:
    print(row)
