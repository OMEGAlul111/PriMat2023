import math
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def iter_logger(iter_count, a, b, table, massx, massy):
    table.add_row([iter_count, a, b, b - a])
    massx.append(iter_count)
    massy.append(b - a)


def dichotomy_method(a, b, eps, func, table):
    massx = []
    massy = []
    func_calls = 0
    delta = eps / 2
    iter_count = 0
    iter_logger(iter_count, a, b, table, massx, massy)
    while b - a > 2 * eps:
        m = (a + b) / 2
        x1 = m - delta
        x2 = m + delta
        func_calls += 2
        if func(x1) < func(x2):
            b = x2
        else:
            a = x1
        iter_count += 1
        iter_logger(iter_count, a, b, table, massx, massy)

    print("Number of iterations:", iter_count)
    print("Number of func calls", func_calls)
    plt.plot(massx, massy)
    return (a + b) / 2


def golden_method(a, b, eps, func, table):
    massx = []
    massy = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    func_calls = 0
    iter_count = 0
    iter_logger(iter_count, a, b, table, massx, massy)
    m = (a + b) / 2
    x1 = b - (b - a) / golden_ratio
    x2 = a + (b - a) / golden_ratio
    f1 = func(x1)
    f2 = func(x2)
    func_calls += 2
    while abs(b - a) > 2 * eps:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / golden_ratio
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / golden_ratio
            f2 = func(x2)
        func_calls += 1
        iter_count += 1
        iter_logger(iter_count, a, b, table, massx, massy)
    print("Number of iterations:", iter_count)
    print("Number of func calls", func_calls)
    plt.plot(massx, massy)
    return (x1 + x2) / 2


def fibb_method(a, b, total_iter, func, table):
    massx = []
    massy = []
    total_iter += 1
    func_calls = 0
    fibb = [1, 1]
    for i in range(total_iter):
        fibb.append(fibb[i] + fibb[i + 1])
    iter_count = 0
    iter_logger(iter_count, a, b, table, massx, massy)
    n = total_iter
    x1 = a + (b - a) * fibb[n - 1] / fibb[n + 1]
    x2 = a + (b - a) * fibb[n] / fibb[n + 1]
    f1 = func(x1)
    f2 = func(x2)
    func_calls += 2
    while n > 1:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * fibb[n - 2] / fibb[n]
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * fibb[n - 1] / fibb[n]
            f2 = func(x2)
        func_calls += 1
        n -= 1
        iter_count += 1
        iter_logger(iter_count, a, b, table, massx, massy)
    print("Number of iterations:", iter_count)
    print("Number of func calls", func_calls)
    plt.plot(massx, massy)
    return (x1 + x2) / 2


def get_u(x1, x2, x3, f1, f2, f3):
    if 2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)) == 0:
        return None
    return x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
                2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)))


def parabola_method(a, b, eps, func, table):
    massx = []
    massy = []
    func_calls = 0
    iter_count = 0
    iter_logger(iter_count, a, b, table, massx, massy)
    x1 = a
    x2 = (a + b) / 2
    x3 = b
    f1 = func(x1)
    f2 = func(x2)
    f3 = func(x3)
    func_calls += 3
    while x3 - x1 > 2 * eps:
        iter_count += 1
        u = get_u(x1, x2, x3, f1, f2, f3)
        fu = func(u)
        func_calls += 1
        if x2 < u:
            if f2 < fu:
                x3 = u
                f3 = fu
            else:
                x1 = x2
                x2 = u
                f1 = f2
                f2 = fu
        else:
            if f2 < fu:
                x1 = u
                f1 = fu
            else:
                x3 = x2
                x2 = u
                f3 = f2
                f2 = fu
        iter_logger(iter_count, x1, x3, table, massx, massy)
    print("Number of iterations:", iter_count)
    print("Number of func calls", func_calls)
    plt.plot(massx, massy)
    return (x1 + x3) / 2


def brent_method(a, b, eps, func, table):
    massx = []
    massy = []
    func_calls = 0
    iter_count = 0
    golden_ratio = (3 - math.sqrt(5)) / 2
    iter_logger(iter_count, a, b, table, massx, massy)
    x = (a + b) / 2
    v = (a + b) / 2
    w = (a + b) / 2
    fx = func(x)
    fv = fx
    fw = fx
    func_calls += 1
    step = b - a
    while (b - a) > 2 * eps:
        p_step = step
        iter_count += 1
        u_parabola = get_u(x, w, v, fx, fw, fv)
        if u_parabola is not None and a + eps < u_parabola < b - eps and np.abs(u_parabola - x) < p_step / 2:
            u = u_parabola
            step = np.abs(u - x)
        else:
            if x < (a + b) / 2:
                u = x + golden_ratio * (b - x)
                step = b - x
            else:
                u = x - golden_ratio * (x - a)
                step = x - a

        fu = func(u)
        func_calls += 1
        if fu < fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if w == x or fu < fw:
                v = w
                w = u
                fv = fw
                fw = fu
            elif v == w or v == x or fu <= fv:
                v = u
                fv = fu
        iter_logger(iter_count, a, b, table, massx, massy)
    print("Number of iterations:", iter_count)
    print("Number of func calls", func_calls)
    plt.plot(massx, massy)
    return (a + b) / 2


def func1(x):
    return math.exp(-x ** 2) * (x ** 2) + (1 - np.e - x ** 2) * math.sin(x)


table = PrettyTable()
table.field_names = ["Iter num", "a", "b", "Length"]
plt.xlabel("Number of iterations")
plt.ylabel("Length")
brent_method(2.15, 2.2, 0.000001, func1, table)
plt.show()
print(table)
