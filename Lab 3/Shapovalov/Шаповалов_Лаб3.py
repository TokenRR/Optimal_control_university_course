import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


PRECISION = 3
np.set_printoptions(precision=PRECISION)


K = 100
alpha = 0
beta = 1
a1 = 0
b1 = np.sinh(1)
a2 = 0
b2 = np.sinh(1)


def get_f(x, dt, a1, b1, a2, b2):
    x1 = np.concatenate(([a1], x[:len(x) // 2], [b1]))
    x2 = np.concatenate(([a2], x[len(x) // 2:], [b2]))

    f = 0
    for i in range(0, len(x1)-1):
        f += (x1[i]**2 + x2[i]**2 + 2*(x1[i+1]-x1[i])/dt*(x2[i+1]-x2[i])/dt)
    return f * dt


def getX1(t):
    return np.sinh(t)


def getX2(t):
    return np.sinh(t)


def plot_e():
    e_t = [i for i in range(10, 55, 10)]
    e1_list = list()
    e2_list = list()
    for e in e_t:
        dt = (beta - alpha) / e
        t = np.linspace(alpha, beta, e)
        initial_X = np.zeros(2 * e - 4)
        result = minimize(get_f, initial_X, args=(dt, a1, b1, a2, b2))
        x = result.x
        x1 = np.concatenate(([a1], x[:len(x) // 2], [b1]))
        x2 = np.concatenate(([a2], x[len(x) // 2:], [b2]))
        e1 = max(abs(x1 - [getX1(ti) for ti in t]))
        e2 = max(abs(x2 - [getX2(ti) for ti in t]))
        e1_list.append(e1)
        e2_list.append(e2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Похибки змінних X1 та X2 в залежності від N')

    axs[0].plot(e_t, e1_list, label='Похибка', color='black')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('e', rotation=0)
    axs[0].legend()
    axs[0].set_title('Похибка Х1')
    axs[0].grid(color='lightgrey')

    axs[1].plot(e_t, e2_list, label='Похибка', color='black')
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('e', rotation=0)
    axs[1].legend()
    axs[1].set_title('Похибка Х2')
    axs[1].grid(color='lightgrey')

    plt.tight_layout()
    plt.show()


def var_24(N):
    initial_X = np.zeros(2*N - 4)
    result = minimize(get_f, initial_X, args=(dt, a1, b1, a2, b2))
    x = result.x
    print(f'\n\nN = {N}, Значення функції в мінімумі = {np.round(result.fun, PRECISION)}')
    
    T = np.linspace(alpha, beta, K)
    X1 = [getX1(ti) for ti in T]
    X2 = [getX2(ti) for ti in T]

    x1 = np.concatenate(([a1], x[:len(x) // 2], [b1]))
    x2 = np.concatenate(([a2], x[len(x) // 2:], [b2]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'N = {N}')

    axs[0].plot(T, X1, label='X1', color='red')
    axs[0].plot(t, x1, label='x1', color='black', linestyle='dashed')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('X', rotation=0)
    axs[0].legend()
    axs[0].grid(color='lightgrey')

    axs[1].plot(T, X2, label='X2', color='red')
    axs[1].plot(t, x2, label='x2', color='black', linestyle='dashed')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X', rotation=0)
    axs[1].legend()
    axs[1].grid(color='lightgrey')

    e1 = max(abs(x1 - [getX1(ti) for ti in t]))
    e2 = max(abs(x2 - [getX2(ti) for ti in t]))

    print(f'\nПохибка Х1 = {round(e1, PRECISION)}')
    print(f'Похибка Х2 = {round(e2, PRECISION)}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for N in [10, 30, 50]:
        dt = (beta - alpha) / N
        t = np.linspace(alpha, beta, N)
        var_24(N)
    plot_e()
    