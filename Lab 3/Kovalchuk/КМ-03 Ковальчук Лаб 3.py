import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


PRECISION = 4
np.set_printoptions(precision=PRECISION)


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

    f_str = func_str

    f_str = f_str.replace(f'x{N}', str(YB))
    for i in range(len(args), 1, -1):
        f_str = f_str.replace(f"x{i}", str(args[i - 1]))
    f_str = f_str.replace('x1', str(YA))
    return eval(f_str)


def solve(n):
    x0 = np.array([1 for _ in range(n + 2)])
    res = scipy.optimize.minimize(j, x0)
    print(f"Мінімальне значення функції: {round(res.fun, PRECISION)}")

    result = res.x[1:-2]
    result[-1] = YB
    y_list = np.concatenate([[YA], result])
    
    t_plot = np.linspace(A, B, 200)
    analytic = [np.exp(2*(1-t)) for t in t_plot]

    plt.plot(x_list[:-1], y_list, label="Наближений розв'язок", color='red')
    plt.plot(t_plot, analytic, label="Точний розв'язок", color='blue')
    plt.title("Розв'язки задачі варіаційного числення")
    plt.legend()
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(c='lightgrey')
    plt.xlabel('t')
    plt.ylabel('x(t)', rotation=0, labelpad=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 10             #  Число точок апроксимації
    A = 0              #  Нижня межа інтегрування
    B = 1              #  Верхня межа інтегрування
    YA = np.exp(2)     #  x(0) = e^2
    YB = 1             #  x(1) = 1
    FUNC = "x'**2+4*x**2"

    step = (B-A)/N
    solve(N)
    
