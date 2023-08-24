import math
import numpy as np
import matplotlib.pyplot as plt


def func1(x):
    return x * np.sin(x)


def true_integral_1(a, b):
    left = np.sin(a) - a * np.cos(a)
    right = np.sin(b) - b * np.cos(b)
    return right-left


def func2(x):
    return x+np.exp(x)


def true_integral_2(a, b):
    left = a*a/2 + np.exp(a)
    right = b*b/2 + np.exp(b)
    return right-left


def rectangle_left_formula(a, b, steps, func):
    result = 0
    h = (b-a)/steps
    for i in range(steps):
        x = a+(b-a)*h*i
        result += h*func(x)
    return result


def rectangle_right_formula(a, b, steps, func):
    result = 0
    h = (b-a)/steps
    for i in range(steps):
        x = a+(b-a)*h*i
        result += h*func(x+h)
    return result


def rectangle_central_formula(a, b, steps, func):
    result = 0
    h = (b-a)/steps
    for i in range(steps):
        x = a+(b-a)*h*i
        result += h*func(x+h/2)
    return result


def trapezoid_formula(a, b, steps, func):
    result = 0
    h = (b - a) / steps
    for i in range(steps):
        x = a + (b - a) * h * i
        result += h / 2 * (func(x) + func(x+h))
    return result


def simpson_formula(a, b, steps, func):
    result = 0
    h = (b - a) / steps
    for i in range(steps):
        x = a + (b - a) * h * i
        result += h / 6 * (func(x) + func(x + h) + 4*func(x+h/2))
    return result


def draw_display_otkl(steps_begin, steps_count, func, func_true_integral, integral_formula):
    steps_list = []
    otkl_list = []
    for i in range(steps_count):
        steps = steps_begin*2**i
        print(steps)
        print("Отклонение при числе шагов", steps, ":", np.abs(integral_formula(0, 1, steps, func)-func_true_integral(0, 1)))
        otkl_list.append(np.abs(integral_formula(0, 1, steps, func)-func_true_integral(0, 1)))
        steps_list.append(steps)
    plt.xlabel("Число шагов")
    plt.ylabel("Отклонение")
    plt.plot(steps_list, otkl_list)
    plt.show()


#вычисление значений зависимости отклонения от числа шагов, рисование графика
#Передается: начальное число шагов, число различных значений шагов, функция, аналитический интеграл, формула численного интегрирования
draw_display_otkl(10, 5, func2, true_integral_2, simpson_formula)