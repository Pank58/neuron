# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

import pandas as pd  # библиотека pandas для работы с данными
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import numpy as np  # numpy для работы с векторами и матрицами

# Считываем данные
df = pd.read_csv('iris.csv')

# Смотрим на данные
print(df.head())

# Три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# Выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# Преобразуем строки в численные метки для трех классов
y = np.where(y == "Iris-setosa", 0, np.where(y == "Iris-versicolor", 1, 2))

# Возьмем два признака для удобства визуализации задачи
X = df.iloc[:, [0, 2, 3]].values

# Признаки в X, ответы в y - посмотрим на плоскости, как выглядит задача
fi = plt.figure(figsize=(8, 6))
aX = fi.add_subplot(111, projection='3d')
X = (X - X.mean(axis=0)) / X.std(axis=0)
aX.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], color='red', marker='o', label='Setosa')
aX.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='blue', marker='x', label='Versicolor')
aX.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], color='green', marker='^', label='Virginica')
aX.set_xlabel('Признак 0')
aX.set_ylabel('Признак 2')
aX.set_zlabel('Признак 3')
aX.legend()
plt.show()

# Функция нейрона для классификации одного класса (one-vs-all)
def neuron(w, x):
    # Вычисляем предсказание
    return np.dot(w[1:], x) + w[0]

# Обучение нейрона для каждого из классов
def train_neuron(X, y, num_classes, eta=0.01, epochs=1000):
    # Инициализация весов для каждого класса
    weights = np.random.random((num_classes, X.shape[1] + 1))  # +1 для смещения
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            for class_idx in range(num_classes):
                # Применяем нейрон для текущего класса
                predict = neuron(weights[class_idx], xi)
                # Если пример принадлежит этому классу, корректируем веса
                if class_idx == target:
                    weights[class_idx, 1:] += eta * (1 - predict) * xi  # Обновляем веса
                    weights[class_idx, 0] += eta * (1 - predict)  # Обновляем смещение
                else:
                    weights[class_idx, 1:] -= eta * predict * xi  # Обновляем веса для других классов
                    weights[class_idx, 0] -= eta * predict  # Обновляем смещение для других классов
    return weights

# Тренируем нейроны для трех классов
num_classes = 3
weights = train_neuron(X, y, num_classes)

# Функция для прогнозирования класса
def predict(X, weights):
    predictions = []
    for xi in X:
        scores = [neuron(w, xi) for w in weights]  # Прогнозируем для каждого нейрона
        predictions.append(np.argmax(scores))  # Выбираем класс с наибольшим результатом
    return np.array(predictions)

# Прогнозируем для всей выборки
predictions = predict(X, weights)

# Подсчитаем количество ошибок
errors = np.sum(predictions != y)
accuracy = 1 - (errors / len(y))  # Точность = 1 - доля ошибок
print(f"Всего ошибок: {errors}")
print(f"Точность модели: {accuracy:.2f}")

# === 3D визуализация ===
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
zz = -(weights[0, 1] * xx + weights[0, 2] * yy + weights[0, 0]) / weights[0, 3]

settings = [
    ("1. Вид косой?", 30, -60),
    ("2. Вид сверху", 90, -90),
    ("3. Вид справа", 0, 0),
    ("4. Вид спереди", 0, -90)
]

# Визуализация каждого из классов
for title, elev, azim in settings:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], color='red', marker='o', label='Setosa')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='blue', marker='x', label='Versicolor')
    ax.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], color='green', marker='^', label='Virginica')

    ax.set_xlabel('Признак 0')
    ax.set_ylabel('Признак 2')
    ax.set_zlabel('Признак 3')
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)

    plt.show()
