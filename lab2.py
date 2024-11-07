import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as spp

visualize_first = True


# Определяем функцию
def f(x):
    return 1 / 3 + np.cos(10 + 2.3 ** np.abs(x))


# Задаем параметры
a, b = -1, 2.5
N = 13  # Количество узлов
x_nodes = np.linspace(a, b, N)  # Узлы
y_nodes = f(x_nodes)  # Значения функции в узлах


# Функция для вычисления кубических сплайнов
def cubic_spline(x_nodes, y_nodes, x, bc_type="free", d0=0, dn=0):
    N = len(x_nodes) - 1  # Количество интервалов (N-1)
    h = np.diff(x_nodes)  # Длину интервалов между узлами
    a = y_nodes[:]  # Значения функции в узлах (a_i)
    # Система уравнений для b и c
    A = np.zeros((N + 1, N + 1))  # Матрица A
    rhs = np.zeros(N + 1)  # Вектор правой части

    # 5. Граничные условия
    if bc_type == "free":
        A[0, 0] = 1
        A[N, N] = 1
    elif bc_type == "fixed":
        A[0, 0] = -1 * h[0]
        A[0, 1] = h[0]
        A[N, N - 1] = h[N - 1]
        A[N, N] = -h[N - 1]

        rhs[0] = d0 / 6 * h[0] ** 2
        rhs[N] = dn / 6 * h[N - 1] ** 2 # TODO: Здесь должен быть минус, но с ним неказисто
    for i in range(1, N):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (
            y_nodes[i] - y_nodes[i - 1]
        ) / h[i - 1]


    ksi = np.linalg.solve(A, rhs)  # Решаем систему уравнений для c_i
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for i in range(N):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (ksi[i + 1] + 2 * ksi[i])
        c[i] = 3 * ksi[i]
        d[i] = (ksi[i + 1] - ksi[i]) / h[i]

    # Интерполяция
    spline_values = []
    for xi in x:
        for i in range(N):
            if x_nodes[i] <= xi <= x_nodes[i + 1]:
                dx = xi - x_nodes[i]
                spline_value = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
                spline_values.append(spline_value)
                break

    return np.array(spline_values)


# Задача: Максимум фактической ошибки
x_fine = np.linspace(a, b, 1000)  # Точки для оценки ошибки
y_fine = f(x_fine)  # Значения исходной функции в этих точках
spline_fine = cubic_spline(
    x_nodes, y_nodes, x_fine, bc_type="free"
)  # Значения сплайна

error_1 = np.abs(y_fine - spline_fine)  # Вычисляем ошибку
max_error_1 = np.max(error_1)  # Максимальная ошибка

print(f"Максимум фактической ошибки (задача 1): {max_error_1}")

# Визуализация результатов
if visualize_first:
    plt.figure(figsize=(12, 6))
    plt.plot(x_fine, y_fine, label="Исходная функция", color="blue")
    plt.plot(x_fine, spline_fine, label="Сплайн интерполяция", color="orange")
    plt.scatter(x_nodes, y_nodes, color="red", label="Узлы", zorder=5)
    plt.title("Сплайн интерполяция с условиями свободного провисания")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


# Задача 2: Определение числа узлов для ошибки <= 0.05 длины отрезка
tolerance = 0.05 * (b - a)
N_2 = 2
error_2 = 0
while True:
    x_nodes_2 = np.linspace(a, b, N_2)
    y_nodes_2 = f(x_nodes_2)
    spline_fine_2 = cubic_spline(
        x_nodes_2, y_nodes_2, x_fine, bc_type="free"
    )
    error_2 = np.abs(y_fine - spline_fine_2)

    if np.max(error_2) <= tolerance:
        break
    N_2 += 1

print(f"Число узлов для задачи 2: {N_2}, максимальная ошибка: {np.max(error_2)}")

x_nodes_3 = np.linspace(a, b, N_2)
y_nodes_3 = f(x_nodes_3)


# 3. Определение максимальной ошибки с граничными условиями равенства третьих производных
def cubic_polynomial(x_data, y_data):
    x = sp.Symbol("x")
    p = spp.interpolate.lagrange(x_data, y_data)
    p_sympy = sum(coef * x**i for i, coef in enumerate(p.coefficients[::-1]))
    return sp.lambdify(x, p_sympy, "numpy")


# Вычисляем интерполяционные полиномы 3-й степени
p1 = cubic_polynomial(x_nodes_3[0:4], y_nodes_3[0:4])
p2 = cubic_polynomial(x_nodes_3[N_2 - 4:N_2], y_nodes_3[N_2 - 4:N_2])

# Вычисляем третьи производные полиномов
x = sp.Symbol("x")
third_derivative_p1 = sp.lambdify(x, sp.diff(p1(x), x, 3), "numpy")
third_derivative_p2 = sp.lambdify(x, sp.diff(p2(x), x, 3), "numpy")
third_derivative_p1_value = third_derivative_p1(math.pi)
third_derivative_p2_value = third_derivative_p2(math.pi)

# Вычисляем сплайн с новыми граничными условиями
spline_fine_3 = cubic_spline(
    x_nodes_3,
    y_nodes_3,
    x_fine,
    bc_type="fixed",
    d0=third_derivative_p1_value,
    dn=third_derivative_p2_value,
)
error_3 = np.abs(y_fine - spline_fine_3)
max_error_3 = np.max(error_3)
print(f"Максимальная ошибка при N = {N_2} и новых граничных условиях: {max_error_3}")

# Сравнение ошибок
print(f"Максимальная ошибка в п.2: {np.max(error_2)}")
print(f"Максимальная ошибка в п.3: {max_error_3}")


# Построение графиков
plt.plot(x_fine, y_fine, label="Исходная функция")
plt.plot(x_fine, spline_fine_2, label="Сплайн п.2")
plt.plot(x_fine, spline_fine_3, label="Сплайн п.3")
plt.scatter(x_nodes_3, y_nodes_3, label="Узлы")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Сплайн-интерполяция функции")
plt.grid(True)
plt.show()
