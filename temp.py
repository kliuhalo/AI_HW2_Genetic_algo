import random

input = []
foo = []

for x in range(100):
    for y in range(100):
        #foo.append(random.randint(0, 10))
        if x < 50:
            foo.append(y)
        elif x == 50:
            foo.append(200)
        else:
            foo.append(100-y)
    input.append(foo)
    foo = []

print(input)

