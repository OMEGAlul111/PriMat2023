import math
import random
import numpy as np
import matplotlib.pyplot as plt


global f_calls
global g_calls


def config_gen(n, k):
    result = []
    curr = 1
    for i in range(n):
        result.append(curr)
        curr = random.randint(2, k * k - 1)
        if i == n - 1:
            result[i] = k * k
        result[i] /= k*k
    return result


def func1(xarr):
    global f_calls
    f_calls += 1
    return xarr[0] * xarr[0] + xarr[1] * xarr[1]


def func2(xarr):
    global f_calls
    f_calls += 1
    return xarr[0] * xarr[0] + 10 * xarr[1] * xarr[1]


def func_big(coeff_arr, xarr):
    global f_calls
    result = 0
    f_calls += 1
    for i in range(len(xarr)):
        result += coeff_arr[i] * xarr[i] ** 2
    return result


def grad1(xarr):
    global g_calls
    g_calls += 1
    return np.array([2 * xarr[0], 2 * xarr[1]])


def grad2(xarr):
    global g_calls
    g_calls += 1
    return np.array([2 * xarr[0], 20 * xarr[1]])


def grad_big(coeff_arr, xarr):
    global g_calls
    result = []
    g_calls += 1
    for i in range(len(xarr)):
        result.append(2 * coeff_arr[i] * xarr[i])
    return np.array(result)


def distance(xarr, yarr):
    result = 0
    for i in range(len(xarr)):
        result += (xarr[i] - yarr[i]) ** 2
    return np.sqrt(result)


def module(xarr):
    result = 0
    for i in range(len(xarr)):
        result += xarr[i] ** 2
    return np.sqrt(result)


def get_u(x1, x2, x3, f1, f2, f3):
    if 2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)) == 0:
        return None
    return x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
            2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)))


def brent_method(a, b, eps, func):
    func_calls = 0
    iter_count = 0
    golden_ratio = (3 - math.sqrt(5)) / 2
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
    return (a + b) / 2


def gradient_descent_const(x0, grad_f, step, eps):
    global f_calls
    global g_calls
    f_calls = 0
    g_calls = 0
    descent_log = [x0]
    dist = 100
    xprev = x0
    iter_count = 0
    while dist > eps:
        iter_count += 1
        xnew = xprev - step * grad_f(xprev)
        dist = distance(xprev, xnew)
        descent_log.append(xnew)
        xprev = xnew
    print("Gradient Descent with constant step complete")
    print("Iterations", iter_count)
    print("Func calls", f_calls)
    print("Grad calls", g_calls)
    return descent_log


def gradient_descent_crushing(x0, func, grad_f, step, eps_armiho, delta, eps_exact):
    global f_calls
    global g_calls
    f_calls = 0
    g_calls = 0
    descent_log = [x0]
    dist = 100
    xprev = x0
    iter_count = 0
    while dist > eps_exact:
        iter_count += 1
        grad_curr = grad_f(xprev)
        fdiff = func(xprev - step * grad_curr) - func(xprev)
        if fdiff > -eps_armiho * step * module(grad_curr):
            step = step * delta
        xnew = xprev - step * grad_curr
        dist = distance(xprev, xnew)
        descent_log.append(xnew)
        xprev = xnew
    print("Gradient Descent with crushing step complete")
    print("Iterations", iter_count)
    print("Func calls", f_calls)
    print("Grad calls", g_calls)
    return descent_log

def gradient_descent_fastest(x0, func, grad_f, eps):
    global f_calls
    global g_calls
    f_calls = 0
    g_calls = 0
    descent_log = [x0]
    dist = 100
    xprev = x0
    iter_count = 0
    while dist > eps:
        iter_count += 1
        grad_curr = grad_f(xprev)
        step = brent_method(0, 10, eps, lambda step: func(xprev - step * grad_curr))
        xnew = xprev - step * grad_curr
        dist = distance(xprev, xnew)
        descent_log.append(xnew)
        xprev = xnew
    print("Gradient Descent with fastest step complete")
    print("Iterations", iter_count)
    print("Func calls", f_calls)
    print("Grad calls", g_calls)
    return descent_log


def gradient_descent_conj(x0, func, grad_f, eps):
    global f_calls
    global g_calls
    f_calls = 0
    g_calls = 0
    descent_log = [x0]
    dist = 100
    xprev = x0
    grad_prev = grad_f(xprev)
    s = -grad_prev
    iter_count = 0
    while dist > eps:
        iter_count += 1
        step = brent_method(0, 10, eps, lambda step: func(xprev + step * s))
        xnew = xprev + step * s
        grad_cur = grad_f(xnew)
        omega = (module(grad_cur)**2) / (module(grad_prev)**2)
        s = -grad_cur + omega * s
        # print("KEK", omega, module(grad_cur), module(grad_prev), grad_cur, grad_prev, xnew, xprev, step)
        grad_prev = grad_cur
        dist = distance(xprev, xnew)
        xprev = xnew
        descent_log.append(xnew)
    print("Gradient Descent with conjugate gradient complete")
    print("Iterations", iter_count)
    print("Func calls", f_calls)
    print("Grad calls", g_calls)
    return descent_log


def make_descent_plot(func, descent_log, plot_size):
    plt.xlabel("X")
    plt.ylabel("Y")
    x = np.linspace(-plot_size, plot_size, 1000)
    y = np.linspace(-plot_size, plot_size, 1000)
    xx, yy = np.meshgrid(x, y)
    xarr = [xx, yy]
    z = func(xarr)
    plt.contour(xx, yy, z)
    xarr = []
    yarr = []
    for i in range(len(descent_log)):
        xarr.append(descent_log[i][0])
        yarr.append(descent_log[i][1])
        if i == 0:
            plt.scatter(xarr[i], yarr[i], marker='x', s=200, color='red')
    plt.plot(xarr, yarr, marker='o', color='black')
    plt.show()


#
#
# 1ST PART OF TASK
#
#

# START POINT:

startp = np.array([2000, 1000])

# UNCOMMENT METHOD YOU WANT TO TEST:

# descent_log = gradient_descent_const(startp, grad2, 0.01, 0.00001)
descent_log = gradient_descent_crushing(startp, func2, grad2, 0.9, 0.9, 0.9, 0.00001)
# descent_log = gradient_descent_fastest(startp, func2, grad2, 0.00001)
# descent_log = gradient_descent_conj(startp, func2, grad2, 0.00001)

# MAKING PLOT:
print(descent_log)

make_descent_plot(func1, descent_log, 10)

#
#
# 2ND PART OF TASK
#
#


# n, k FOR GENERATION:

n = 10
k = 10

# ITERATIONS FOR TEST:

iter_count = 10

# TESTING...

fcall_avg=0
gcall_avg=0
for i in range(iter_count):
    # GENERATING FUNCTION:

    coeff_gen = config_gen(n, k)
    startp = []
    for j in range(n):
        startp.append(10)
    startp = np.array(startp)

    # UNCOMMENT METHOD U WANT TO TEST:

    # descent_log = gradient_descent_const(startp, lambda xarr: grad_big(coeff_gen, xarr), 0.1, 0.0001)
    # descent_log = gradient_descent_crushing(startp, lambda xarr: func_big(coeff_gen, xarr),
    #                                        lambda xarr: grad_big(coeff_gen, xarr), 0.99, 0.99, 0.99, 0.00001)
    descent_log = gradient_descent_fastest(startp, lambda xarr: func_big(coeff_gen, xarr), lambda xarr: grad_big(coeff_gen, xarr), 0.00001)
    # descent_log = gradient_descent_conj(startp, lambda xarr: func_big(coeff_gen, xarr), lambda xarr: grad_big(coeff_gen, xarr), 0.00001)

    fcall_avg += f_calls
    gcall_avg += g_calls

print("FUNC CALLS AVG:", fcall_avg/iter_count)
print("GRAD CALLS AVG:", gcall_avg/iter_count)
