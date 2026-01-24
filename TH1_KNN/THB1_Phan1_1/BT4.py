n = int(input("Nhap n: "))

def soHoanHao(a):
    sum = 0
    for i in range (1, a):
        if a % i == 0:
            sum += i
    return sum

if n == soHoanHao(n):
    print (n, " la so hoan hao")
else:
    print(n, " khong la so hoan hao")