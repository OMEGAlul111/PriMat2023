import math
import numpy as np
import matplotlib.pyplot as plt


def leftdif(a, b, step, steps, f):
    h = (b-a)/steps
    x = a+(b-a)*h*step
    return (f(x) - f(x - h)) / h


def rightdif(a, b, step, steps, f):
    h = (b-a)/steps
    x = a + (b - a) * h*step
    return (f(x + h) - f(x)) / h


def centraldif(a, b, step, steps, f):
    h = (b-a)/steps
    x = a + (b - a) * h*step
    if step == 0:
        return (-3*f(x)+4*f(x+h)-f(x+2*h))/(2*h)
    if step == steps:
        return (f(x-2*h)-4*f(x-h)+3*f(x))/(2*h)
    return (f(x + h) - f(x - h)) / (2 * h)


def leftsco(a, b, steps, f, fdif):
    sco = 0
    h = (b - a) / steps
    for i in range(steps):
        j = i + 1
        sco += (leftdif(a, b, j, steps, f) - fdif(a + h * j)) ** 2
    return math.sqrt(sco / steps)


def rightsco(a, b, steps, f, fdif):
    sco = 0
    h = (b - a) / steps
    for i in range(steps):
        sco += (rightdif(a, b, i, steps, f) - fdif(a + h * i)) ** 2
    return math.sqrt(sco / steps)


def centralsco(a, b, steps, f, fdif):
    sco = 0
    h = (b - a) / steps
    for i in range(steps + 1):
        sco += (centraldif(a, b, i, steps, f) - fdif(a + h * i)) ** 2
    return math.sqrt(sco / (steps + 1))


def func1(x):
    return x * np.sin(x)


def truedif1(x):
    return np.sin(x) + x * np.cos(x)


def func2(x):
    return x+np.exp(x)


def truedif2(x):
    return 1+np.exp(x)


def draw_diffunc_plt(steps, f, fdif):
    plt.xlabel("x")
    plt.ylabel("f '(x)")
    x = np.arange(0, 1.01, 0.001)
    plt.plot(x, fdif(x))

    x = np.arange(0, 0.9999999, 1/steps)
    y = []
    for i in range(steps):
        y.append(leftdif(0, 1, i+1, steps, f))
    plt.scatter(x, y)

    x = np.arange(1/steps, 1.0000001, 1 / steps)
    y = []
    for i in range(steps):
        y.append(rightdif(0, 1, i, steps, f))
    plt.scatter(x, y)

    x = np.arange(0, 1.0000001, 1 / steps)
    y = []
    for i in range(steps+1):
        y.append(centraldif(0, 1, i, steps, f))
    plt.scatter(x, y)

    plt.show()


def draw_display_sco(steps_begin, steps_count, f, fdif, sco):
    steps_list=[]
    sco_list=[]
    for i in range(steps_count):
        steps = steps_begin*2**i
        print(steps)
        print("СКО при числе шагов", steps, ":", sco(0, 1, steps, f, fdif))
        sco_list.append(sco(0, 1, steps, f, fdif))
        steps_list.append(steps)
    plt.xlabel("Число шагов")
    plt.ylabel("СКО")
    plt.plot(steps_list, sco_list)
    plt.show()


#рисование приближения графика аналитической производной левой, правой и центральной разностными производными
#Передается: число шагов, функция, аналитическая производная
draw_diffunc_plt(10, func2, truedif2)

#вычисление значений зависимости СКО от числа шагов, построения графика этой зависимости
#Передается: начальное число шагов, число различных значений шагов, функция, аналитическая производная, функция вычисления СКО
draw_display_sco(10, 5, func2, truedif2, centralsco)