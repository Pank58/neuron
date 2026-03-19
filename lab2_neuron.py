# # -*- coding: utf-8 -*-
# """
# Created on Mon Feb 22 21:30:32 2025

# @author: AM4
# """

# # Попробуем обучить один нейрон на задачу классификации двух классов

# import pandas as pd # библиотека pandas нужна для работы с данными
# import matplotlib.pyplot as plt # matplotlib для построения графиков
# import numpy as np # numpy для работы с векторами и матрицами

# # Считываем данные 
# # df = pd.read_csv('https://archive.ics.uci.edu/ml/'
# #     'machine-learning-databases/iris/iris.data', header=None)

# df = pd.read_csv('data.csv')


# # смотрим что в них
# print(df.head())

# # три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# # выделим целевую переменную в отдельную переменную
# y = df.iloc[:, 4].values

# # так как ответы у нас строки - нужно перейти к численным значениям
# y = np.where(y == "Iris-setosa", 1, -1)

# # возьмем два признака, чтобы было удобне визуализировать задачу
# X = df.iloc[:, [0, 2]].values

# # Признаки в X, ответы в y - постмотрим на плоскости как выглядит задача
# plt.figure
# plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o')
# plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x')

# # переходим к созданию нейрона
# # функция нейрона:
# # значение = w1*признак1+w2*признак2+w0
# # ответ = 1, если значение > 0
# # ответ = -1, если значение < 0

# def neuron(w,x):
#     if((w[1]*x[0]+w[2]*x[1]+w[0])>=0):
#         predict = 1
#     else: 
#         predict = -1
#     return predict

# # проверим как это работает (веса зададим пока произвольно)
# w = np.array([0, 0.1, 0.4])
# print(neuron(w,X[1])) # вывод ответа нейрона для примера с номером 1


# # теперь создадим процедуру обучения
# # корректировка веса производится по выражению:
# # w_new = w_old + eta*x*y

# # зададим начальные значения весов
# w = np.random.random(3)
# eta = 0.01  # скорость обучения
# w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
# for xi, target, j in zip(X, y, range(X.shape[0])):
#     predict = neuron(w,xi)   
#     w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
#     w[0] += eta * (target - predict)
#     # каждую 10ю итерацию будем сохранять набор весов в специальном списке
#     if(j%10==0):
#         w_iter.append(w.tolist())

# # посчитаем ошибки
# sum_err = 0
# for xi, target in zip(X, y):
#     predict = neuron(w,xi) 
#     sum_err += (target - predict)/2

# print("Всего ошибок: ", sum_err)


# # попробуем визуализировать процесс обучения
# xl=np.linspace(min(X[:,0]), max(X[:,0])) # диапазон координаты x для построения линии

# # построим сначала данные на плоскости
# plt.figure
# plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o')
# plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x') 

# # потом в цикле будем брать набор весов из сохраненного списка и по нему строить линию
# for i,w in zip(range(len(w_iter)), w_iter):
#     yl = -(xl*w[1]+w[0])/w[2] # уравнение линии
#     plt.plot(xl, yl) # строим разделяющую границу
#     plt.text(xl[-1], yl[-1], i, dict(size=10, color='gray')) # подписываем номер линии
#     plt.pause(1)
    
# plt.text(xl[-1]-0.3, yl[-1], 'END', dict(size=14, color='red'))
# plt.show() 
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков

# Считываем данные
df = pd.read_csv('data.csv')

print(df.head())

# Целевая переменная
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)


X = df.iloc[:, [0, 1, 2]].values

def neuron(w, x):
    value = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2]
    return 1 if value >= 0 else -1


w = np.array([0.0, 0.1, 0.2, 0.3])  # смещение + 3 признака
print("Пример работы нейрона:", neuron(w, X[0]))

w = np.random.random(4)
eta = 0.01
w_iter = []

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    error = target - predict
    w[1:] += eta * error * xi  # корректировка весов для признаков
    w[0] += eta * error        # корректировка смещения
    if j % 10 == 0:
        w_iter.append(w.tolist())


sum_err = sum((target - neuron(w, xi)) / 2 for xi, target in zip(X, y))
print("Всего ошибок: ", sum_err)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Строим точки для классов
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x', label='Other')

# Построим разделяющую гиперплоскость
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Для каждого значения x1, x2 на сетке вычислим x3, чтобы линия была разделяющей
x3_grid = -(w[0] + w[1]*x1_grid + w[2]*x2_grid) / w[3]


ax.plot_surface(x1_grid, x2_grid, x3_grid, color='yellow', alpha=0.5)


ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
ax.legend()

plt.show()
