import random

mydict = {}

while len(mydict) < 10:
    key = random.randint(1, 10)
    value = random.randint(1, 10)
    mydict[key] = value

print("Từ điển ban đầu:")
print(mydict)

array_2c = []

for key, value in mydict.items():
    array_2c.append([key, value])

print("\nMảng 2 chiều:")
for row in array_2c:
    print(row)
