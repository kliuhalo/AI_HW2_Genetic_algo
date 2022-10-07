import random

input = []
foo = []

for x in range(15):
    for y in range(15):
        foo.append(random.randint(0, 10))
    input.append(foo)
    foo = []
print(input)