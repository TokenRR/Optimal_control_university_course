import numpy as np
import matplotlib.pyplot as plt


def u(_i, _u0):
    return min(umax, _u0*((_i/N)**delta))


def p(_i, _r0, _c0):
    _r = _r0*np.exp(-gamma*_i/N)
    _c = _c0/((_i/N)**lamda)
    _p = -_r*_i*np.log(1-_c)/N
    return _p


def st(_s, _i, _r0, _c0, _u0):
    return -_s*p(_i, _r0, _c0) - _s*u(_i, _u0)


def it(_s, _i, _r0, _c0):
    return _s*p(_i, _r0, _c0) - alpha*_i - beta*_i


def rt(_s, _i, _u0):
    return alpha*_i + _s*u(_i, _u0)


def dt(_i):
    return beta*_i


def rk4(_s, _i, _r, _d, _h, _r0, _c0, _u0):
    k1 = _h * st(_s, _i, _r0, _c0, _u0)
    q1 = _h * it(_s, _i, _r0, _c0)
    l1 = _h * rt(_s, _i, _u0)
    m1 = _h * dt(_i)
    k2 = _h * st(_s, _i, _r0, _c0, _u0)
    q2 = _h * it(_s, _i, _r0, _c0)
    l2 = _h * rt(_s, _i, _u0)
    m2 = _h * dt(_i)
    k3 = _h * st(_s, _i, _r0, _c0, _u0)
    q3 = _h * it(_s, _i, _r0, _c0)
    l3 = _h * rt(_s, _i, _u0)
    m3 = _h * dt(_i)
    k4 = _h * st(_s, _i, _r0, _c0, _u0)
    q4 = _h * it(_s, _i, _r0, _c0)
    l4 = _h * rt(_s, _i, _u0)
    m4 = _h * dt(_i)
    s_next = _s + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    i_next = _i + (q1 + 2 * q2 + 2 * q3 + q4) / 6
    r_next = _r + (l1 + 2 * l2 + 2 * l3 + l4) / 6
    d_next = _d + (m1 + 2 * m2 + 2 * m3 + m4) / 6
    return s_next, i_next, r_next, d_next


def test_n():
    N = [1_000, 10_000, 100_000, 1_000_000]
    s = [900, 9_000, 90_000, 900_000]
    i = [100, 1_000, 10_000, 100_000]
    s_res = []
    i_res = []
    r_res = []
    d_res = []
    for k in range(len(N)):
        s_list = [s[k]]
        i_list = [i[k]]
        r_list = [r]
        d_list = [d]
        for j in range(step):
            res1, res2, res3, res4 = rk4(s_list[j], i_list[j], r_list[j], d_list[j], h, r0, c0, u0)
            s_list.append(res1)
            i_list.append(res2)
            r_list.append(res3)
            d_list.append(res4)

        s_list = np.array(s_list)
        i_list = np.array(i_list)
        r_list = np.array(r_list)
        d_list = np.array(d_list)
        s_res.append(s_list)
        i_res.append(i_list)
        r_res.append(r_list)
        d_res.append(d_list)

    x = np.arange(t0, t + h, h)
    plt.subplot(2, 2, 1)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[0], color='blue', label="Susceptibles")
    plt.plot(x, i_res[0], color='red', label="Infected")
    plt.plot(x, r_res[0], color='green', label="Recovered")
    plt.plot(x, d_res[0], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')

    plt.subplot(2, 2, 2)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[1], color='blue', label="Susceptibles")
    plt.plot(x, i_res[1], color='red', label="Infected")
    plt.plot(x, r_res[1], color='green', label="Recovered")
    plt.plot(x, d_res[1], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')

    plt.subplot(2, 2, 3)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[2], color='blue', label="Susceptibles")
    plt.plot(x, i_res[2], color='red', label="Infected")
    plt.plot(x, r_res[2], color='green', label="Recovered")
    plt.plot(x, d_res[2], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')

    plt.subplot(2, 2, 4)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[3], color='blue', label="Susceptibles")
    plt.plot(x, i_res[3], color='red', label="Infected")
    plt.plot(x, r_res[3], color='green', label="Recovered")
    plt.plot(x, d_res[3], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()
    plt.show()


def test_r():
    r_test = [1, 15, 30, 50]
    s_res = []
    i_res = []
    r_res = []
    d_res = []
    for k in range(len(r_test)):
        s_list = [s]
        i_list = [i]
        r_list = [r]
        d_list = [d]
        for j in range(step):
            res1, res2, res3, res4 = rk4(s_list[j], i_list[j], r_list[j], d_list[j], h, r_test[k], c0, u0)
            s_list.append(res1)
            i_list.append(res2)
            r_list.append(res3)
            d_list.append(res4)

        s_list = np.array(s_list)
        i_list = np.array(i_list)
        r_list = np.array(r_list)
        d_list = np.array(d_list)
        s_res.append(s_list)
        i_res.append(i_list)
        r_res.append(r_list)
        d_res.append(d_list)

    x = np.arange(t0, t + h, h)
    plt.subplot(2, 2, 1)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[0], color='blue', label="Susceptibles")
    plt.plot(x, i_res[0], color='red', label="Infected")
    plt.plot(x, r_res[0], color='green', label="Recovered")
    plt.plot(x, d_res[0], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[1], color='blue', label="Susceptibles")
    plt.plot(x, i_res[1], color='red', label="Infected")
    plt.plot(x, r_res[1], color='green', label="Recovered")
    plt.plot(x, d_res[1], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[2], color='blue', label="Susceptibles")
    plt.plot(x, i_res[2], color='red', label="Infected")
    plt.plot(x, r_res[2], color='green', label="Recovered")
    plt.plot(x, d_res[2], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[3], color='blue', label="Susceptibles")
    plt.plot(x, i_res[3], color='red', label="Infected")
    plt.plot(x, r_res[3], color='green', label="Recovered")
    plt.plot(x, d_res[3], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()
    plt.show()


def test_c():
    c_test = [0.5, 0.6, 0.7, 0.8]
    s_res = []
    i_res = []
    r_res = []
    d_res = []
    for k in range(len(c_test)):
        s_list = [s]
        i_list = [i]
        r_list = [r]
        d_list = [d]
        for j in range(step):
            res1, res2, res3, res4 = rk4(s_list[j], i_list[j], r_list[j], d_list[j], h, r0, c_test[k], u0)
            s_list.append(res1)
            i_list.append(res2)
            r_list.append(res3)
            d_list.append(res4)

        s_list = np.array(s_list)
        i_list = np.array(i_list)
        r_list = np.array(r_list)
        d_list = np.array(d_list)
        s_res.append(s_list)
        i_res.append(i_list)
        r_res.append(r_list)
        d_res.append(d_list)

    x = np.arange(t0, t + h, h)
    plt.subplot(2, 2, 1)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[0], color='blue', label="Susceptibles")
    plt.plot(x, i_res[0], color='red', label="Infected")
    plt.plot(x, r_res[0], color='green', label="Recovered")
    plt.plot(x, d_res[0], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[1], color='blue', label="Susceptibles")
    plt.plot(x, i_res[1], color='red', label="Infected")
    plt.plot(x, r_res[1], color='green', label="Recovered")
    plt.plot(x, d_res[1], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[2], color='blue', label="Susceptibles")
    plt.plot(x, i_res[2], color='red', label="Infected")
    plt.plot(x, r_res[2], color='green', label="Recovered")
    plt.plot(x, d_res[2], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[3], color='blue', label="Susceptibles")
    plt.plot(x, i_res[3], color='red', label="Infected")
    plt.plot(x, r_res[3], color='green', label="Recovered")
    plt.plot(x, d_res[3], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()
    plt.show()


def test_u():
    u_test = [0.1, 0.2, 0.4, 0.8]
    s_res = []
    i_res = []
    r_res = []
    d_res = []
    for k in range(len(u_test)):
        s_list = [s]
        i_list = [i]
        r_list = [r]
        d_list = [d]
        for j in range(step):
            res1, res2, res3, res4 = rk4(s_list[j], i_list[j], r_list[j], d_list[j], h, r0, c0, u_test[k])
            s_list.append(res1)
            i_list.append(res2)
            r_list.append(res3)
            d_list.append(res4)

        s_list = np.array(s_list)
        i_list = np.array(i_list)
        r_list = np.array(r_list)
        d_list = np.array(d_list)
        s_res.append(s_list)
        i_res.append(i_list)
        r_res.append(r_list)
        d_res.append(d_list)

    x = np.arange(t0, t + h, h)
    plt.subplot(2, 2, 1)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[0], color='blue', label="Susceptibles")
    plt.plot(x, i_res[0], color='red', label="Infected")
    plt.plot(x, r_res[0], color='green', label="Recovered")
    plt.plot(x, d_res[0], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[1], color='blue', label="Susceptibles")
    plt.plot(x, i_res[1], color='red', label="Infected")
    plt.plot(x, r_res[1], color='green', label="Recovered")
    plt.plot(x, d_res[1], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[2], color='blue', label="Susceptibles")
    plt.plot(x, i_res[2], color='red', label="Infected")
    plt.plot(x, r_res[2], color='green', label="Recovered")
    plt.plot(x, d_res[2], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.title('Solution SIR model')
    plt.plot(x, s_res[3], color='blue', label="Susceptibles")
    plt.plot(x, i_res[3], color='red', label="Infected")
    plt.plot(x, r_res[3], color='green', label="Recovered")
    plt.plot(x, d_res[3], color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solutions')
    plt.tight_layout()
    plt.show()


def first():
    s_list = [s]
    i_list = [i]
    r_list = [r]
    d_list = [d]
    for j in range(step):
        res1, res2, res3, res4 = rk4(s_list[j], i_list[j], r_list[j], d_list[j], h, r0, c0, u0)
        s_list.append(res1)
        i_list.append(res2)
        r_list.append(res3)
        d_list.append(res4)
    s_list = np.array(s_list)
    i_list = np.array(i_list)
    r_list = np.array(r_list)
    d_list = np.array(d_list)
    max_infected = max(i_list)
    max_time = t0 + h*(np.where(i_list == max_infected)[0])
    print('\nПік інфікованих = ', int(max_infected), 'у', int(round(max_time[0], 0)), 'день')
    end_time = t0 + h*np.where(i_list < 1)[0][0]
    print("Кількість suspected індивідів: ", int(s_list[-1]))
    print("Кількість infected  індивідів: ", int(i_list[-1]))
    print("Кількість recovered індивідів: ", int(r_list[-1]))
    print("Кількість dead      індивідів: ", int(d_list[-1]))
    print("Кількість днів, які тривала епідемія: ", int(end_time))
    x = np.arange(t0, t+h, h)
    plt.title('Solution SIR model')
    plt.plot(x, s_list, color='blue', label="Susceptibles")
    plt.plot(x, i_list, color='red', label="Infected")
    plt.plot(x, r_list, color='green', label="Recovered")
    plt.plot(x, d_list, color='black', label="Dead")
    plt.legend()
    plt.grid(c='lightgrey')
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.xlabel('Days')
    plt.ylabel('Solution')
    # plt.xlim(0, int(end_time)+1)
    plt.xticks(np.arange(0, t, 5))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    N = 10_000
    s = 9_500
    i = 500
    r = 0
    d = 0
    c0 = 0.5
    r0 = 10
    u0 = 0.1
    alpha = 0.1
    beta = 0.05
    t0, t = 0, 70

    h = 0.1
    step = int((t-t0)/h)

    gamma = 6
    delta = 0.1
    lamda = 0.01
    umax = 0.9
    
    # first()

    # test_n()

    # test_r()

    # test_c()

    test_u()
