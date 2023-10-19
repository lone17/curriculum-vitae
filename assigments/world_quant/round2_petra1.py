# What happens when you run this Python code?
a, b = 0, 1


def fibonacci(n):
    for _ in range(n):
        a, b = b, a + b

    return a


print(fibonacci(4))
