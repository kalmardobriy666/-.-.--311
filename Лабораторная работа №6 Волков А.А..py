# Задание 1
def fahrenheit_to_kelvin(fahrenheit):
    kelvin = (5/9) * (fahrenheit - 32) + 273.15
    return kelvin

fahrenheit = [-88.6, -29.2, 32.0, 93.2, 129.2, 152.6, 212.0]
for f in fahrenheit:
    print(f"{f} Fahrenheit = {fahrenheit_to_kelvin(f):.2f} Kelvin")

fahrenheit = [-88.6, -29.2, 32.0, 93.2, 129.2, 152.6, 212.0]
kelvin = [234.26, 244.82, 273.15, 307.04, 327.04, 340.37, 373.15]

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8), dpi=50)
plt.scatter(fahrenheit, kelvin, label='входные данные', marker='$f$')
plt.xlabel('fahrenheit')
plt.ylabel('kelvin')
plt.legend()
plt.grid(True)
plt.show()

for f, k in zip(fahrenheit, kelvin):
    print(f'фаренгейт {f} → кельвин {k}')

from sklearn.linear_model import LinearRegression

X = [[f] for f in fahrenheit]
y = kelvin

lr = LinearRegression()
lr.fit(X, y)


fahrenheit = [[-88.6], [-29.2], [32.0], [93.2], [129.2], [152.6], [212.0]]
kelvin = lr.predict(fahrenheit)

kelvin

for f, k in zip(fahrenheit, kelvin):
    print(f'фаренгейт {f[0]} → кельвин {k}')

import numpy as np

x_range = np.arange(-100, 220)
y_range = lr.predict(x_range.reshape(-1, 1))

plt.figure(figsize=(15, 8), dpi=80)
plt.plot(x_range, y_range, label='уравнение')
plt.scatter(fahrenheit, kelvin, label='входные данные')
plt.scatter(
    [f[0] for f in fahrenheit],
    kelvin,
    label='предсказанные значения'
)
plt.xlabel('fahrenheit')
plt.ylabel('kelvin')
plt.legend()
plt.grid(True)
plt.show()

# Задание 2

# Ввел команду в терминал: git clone https://github.com/kalmardobriy666/Volkov-A.A.-IST-311.git

# Задание 3

# 1.Столбчатая диаграмма

import matplotlib.pyplot as plt

subjects = ['Math', 'Physics', 'Informatics', 'English']
scores = [85, 78, 92, 88]

plt.figure()
plt.bar(subjects, scores)
plt.xlabel('Предмет')
plt.ylabel('Баллы')
plt.title('Успеваемость по предметам')
plt.grid(True)
plt.show()

# 2.Гистограмма

import matplotlib.pyplot as plt

values = [2, 3, 3, 5, 7, 7, 7, 8, 9, 10, 10, 10, 10]

plt.figure()
plt.hist(values, bins=5)
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма распределения значений')
plt.grid(True)
plt.show()

# 3.Круговая диаграмма

import matplotlib.pyplot as plt

languages = ['Python', 'C++', 'Java', 'JavaScript']
usage = [40, 25, 20, 15]

plt.figure()
plt.pie(usage, labels=languages, autopct='%1.1f%%')
plt.title('Популярность языков программирования')
plt.show()

# Задание 4

import math

print("Число Эйлера (e):", math.e)
print("Число Пи (π):", math.pi)
print("nan:", math.nan)

n = 7
print(f"Факториал числа {n}:", math.factorial(n))

memory = 128

gcd_value = math.gcd(n, memory)
print(f"НОД чисел {n} и {memory}:", gcd_value)

# Задания повышенной сложности

# Задание 1 (КОТ)

plt.figure(figsize=(6, 6))

# Голова
head = plt.Circle((0, 0), 1.0, fill=False)
plt.gca().add_patch(head)

# Уши
ear_left_x = [-0.6, -0.2, -0.9]
ear_left_y = [0.6, 1.3, 0.6]
plt.plot(ear_left_x, ear_left_y)

ear_right_x = [0.6, 0.2, 0.9]
ear_right_y = [0.6, 1.3, 0.6]
plt.plot(ear_right_x, ear_right_y)

# Глаза
eye_left = plt.Circle((-0.4, 0.2), 0.1)
eye_right = plt.Circle((0.4, 0.2), 0.1)
plt.gca().add_patch(eye_left)
plt.gca().add_patch(eye_right)

# Нос
nose_x = [-0.1, 0.1, 0]
nose_y = [-0.1, -0.1, -0.3]
plt.fill(nose_x, nose_y)

# Рот
t = np.linspace(-np.pi/2, np.pi/2, 100)
plt.plot(0.2 * np.sin(t), -0.35 + 0.15 * np.cos(t))

for y in [-0.2, -0.3, -0.4]:
    plt.plot([-0.2, -1.0], [y, y])
    plt.plot([0.2, 1.0], [y, y])

plt.axis('equal')
plt.axis('off')
plt.show()

# Задание 2

# Используемые библиотеки

# NumPy — используется для работы с массивами данных и числовыми операциями.
# Matplotlib — применяется для визуализации изображений и графиков обучения модели.
# TensorFlow — библиотека для создания, обучения и тестирования нейронных сетей.
# Keras используется как высокоуровневый API для упрощения работы с нейросетями.

# import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    'Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
]

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[y_train == i][0], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)

model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

plt.figure(figsize=(8, 3))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    label = np.argmax(predictions[i])
    plt.title(class_names[label])
    plt.axis('off')

plt.show()



















