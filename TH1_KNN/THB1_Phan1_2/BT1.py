a = []

while True:
    x = input()
    if x == '$':
        break
    else:
        a.append(x)

sum = 0
for i in a:
    sum += int(i)

print("Gia tri trung binh cua cac phan tu: ", sum / len(a))