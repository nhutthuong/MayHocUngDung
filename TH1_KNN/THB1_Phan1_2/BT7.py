# Nhập số nguyên N
N = int(input("Nhập số nguyên N: "))

# Tạo từ điển
mydict = {}

for i in range(1, N+1):
    mydict[i] = i * i

# In từ điển
print("N = ", N)
print("mydict = ", mydict)
