import time

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


PRECISION = 3
np.set_printoptions(precision=PRECISION)


def x(t):
    return -1j*t


def j(args):
    global x_list
    x_list = []
    _b = B + step
    dt = (_b-A)/N

    func_str = ""
    for i in range(1, N + 2):
        x_list.append(A + (i - 1) * dt)
        func_str += "(" + FUNC.replace("x'", f"((y{i + 1}-y{i})/{dt})")\
                              .replace("x", f"(x{i})")\
                              .replace("t", f"{dt}") + f")*{dt}"
        if i != N+1:
            func_str += "+"
    func_str = func_str.replace("y", "x")
    func_str = f"(1/{N})*(" + func_str + ")"

    # Add regularization term
    reg_term = LAMBDA * np.sum(np.square(args))
    func_str += f" + {reg_term}"

    f_str = func_str

    f_str = f_str.replace(f'x{N}', str(YB))
    for i in range(len(args), 1, -1):
        f_str = f_str.replace(f"x{i}", str(args[i - 1]))
    f_str = f_str.replace('x1', str(YA))
    return eval(f_str)


def variant_16(n):
    print(f"Задача варіаційного числення для N = {n} та lmd = {LAMBDA}")
    x0 = np.array([1 for _ in range(n + 2)])
    res = scipy.optimize.minimize(j, x0)
    print(f"Значення мінімізованої функції: {round(res.fun, PRECISION)}")

    result = res.x[1:-2]
    result[-1] = YB
    y_list = np.concatenate([[YA], result])

    t_plot = np.linspace(A, B, n)
    analytic = [np.sqrt(65/16 - (t-7/4)**2) for t in t_plot]

    print(f"Похибка (Mean Squared Error) = {mean_squared_error(analytic, y_list)}")

    plt.plot(x_list[:-1], y_list, label="Наближений розв'язок", color='green')
    plt.plot(t_plot, analytic, label="Точний розв'язок", color='black', linestyle='dashed')

    plt.title(f"Розв'язки задачі варіаційного числення при N = {n}, λ = {LAMBDA}")
    plt.legend()
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(c='lightgrey')
    plt.xlabel('t')
    plt.ylabel('x(t)', rotation=0, labelpad=20)
    plt.tight_layout()
    print(f"Витрачено часу: {time.time() - start_time:.2f} секунд\n")
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    N = 10             #  Число точок апроксимації
    A = 0              #  Нижня межа інтегрування
    B = 1/2            #  Верхня межа інтегрування
    YA = 1             #  x(0) = 1
    YB = np.sqrt(5/2)  #  x(1/2) = sqrt(5/2)
    LAMBDA = 10**(-5)  #  Параметр регуляризації
    FUNC = "x**(-1)*(1+(x')**2)**(1/2)"
    step = (B-A)/N

    print('\n\nЛабораторна робота №3\nРоманецький Микита\nВаріант 16\n')
    step = (B-A)/N
    variant_16(N)
