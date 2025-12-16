# Задание 1
for month in range(1, 13):
    print(month)
# Задание 2
number = int(input("Введите число: "))

if number % 2 == 0:
    print("Четное")
else:
    print("Нечетное")
# Задание 3
N = int(input("Введите число N: "))

if N > 40:
    print("not stonk")
else:
    print("stonk")
 
# Задание 4
def is_leap(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

year = int(input("Введите год: "))
if is_leap(year):
    print("Високосный")
else:
    print("Не високосный")

# Задание 5
def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True

number = int(input("Введите число: "))
if is_prime(number):
    print("Простое")
else:
    print("Не простое")

# Задание 6
a = 10
b = 20

if a > b:
    a, b = b, a
elif b > a * 3.6:
    a = (a - 138) / 2 ** 1.3
    b = ((-69 / 28) ** 5.1) * 4

print("a:", a)
print("b:", b)

# Задание 7
numbers = []
for i in range(5):
    num = int(input(f"Введите число {i+1}: "))
    numbers.append(num)

if len(set(numbers)) == 5:  
    if all(num % 2 == 0 for num in numbers):  
        print("Все числа уникальны и четные")
    elif all(num < 0 for num in numbers):
        print("Все числа уникальны и отрицательные")
    elif 256 in numbers and 1024 in numbers:
        print("Все числа уникальны и есть 256 и 1024")
    else:
        print("Все числа уникальны, но не подходят под другие условия")
else:
    print("Не все числа уникальны")

# Задание 8
import numpy as np
from scipy.optimize import fsolve

def f(x):
    return x**3 + x**2 - 4*x - 12

roots = fsolve(f, [-10, 0, 10]) 

roots = np.sort(roots)

intervals = []
for i in range(len(roots) - 1):
    intervals.append((roots[i], roots[i+1]))

def is_even_or_odd(x):
    if x % 2 == 0:
        return "чётным"
    else:
        return "нечётным"

print("Корни многочлена:", roots)
print("Интервалы, на которых неравенство выполняется:", intervals)

x_value = 5
result = f(x_value)
print(f"Для x = {x_value}, результат неравенства {result} является {is_even_or_odd(result)} числом.")