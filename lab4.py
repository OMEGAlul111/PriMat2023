import math
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg.linalg

global iter_count
iter_count = 0


def gen_hilbert(n):
    result = []
    for i in range(n):
        ii = i+1
        row = []
        for j in range(n):
            jj = j+1
            row.append(1/(ii+jj-1))
        row = np.array(row, dtype=float)
        result.append(row)
    result = np.array(result)
    return result


def gen_random(n, k):
    result = []

    for i in range(n):
        ii = i+1
        row = []
        for j in range(n):
            jj = j+1
            if ii != jj:
                row.append(random.randint(-4, 0))
            else:
                row.append(0)
        row = np.array(row, dtype=float)
        result.append(row)
    result = np.array(result)

    for i in range(n):
        sum = 0
        for j in range(n):
            sum += result[i][j]
        if i == 0:
            result[i][i] = -sum+pow(10, -k)
        else:
            result[i][i] = -sum

    return result


def gen_b(a_matrix):
    b = []
    n = len(a_matrix)
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += a_matrix[i][j]*(j+1)
        b.append(sum)
    return np.array(b)


def distance(xarr, yarr):
    global iter_count
    result = 0
    for i in range(len(xarr)):
        result += (xarr[i] - yarr[i]) ** 2
    iter_count += len(xarr)
    return np.sqrt(result)


def gauss_method(a_matrix, b):
    global iter_count
    n = len(a_matrix)
    for j in range(n):
        max_pos = j
        for i in range(j+1, n):
            iter_count += 1
            if np.abs(a_matrix[i][j]) > np.abs(a_matrix[max_pos][j]):
                max_pos = i
        buf = np.copy(a_matrix[max_pos])
        a_matrix[max_pos] = np.copy(a_matrix[j])
        a_matrix[j] = np.copy(buf)
        buf = b[max_pos]
        b[max_pos]=b[j]
        b[j]=buf

        for i in range(j+1, n):
            coeff = a_matrix[i][j]/a_matrix[j][j]
            for jj in range(j, n):
                iter_count += 1
                a_matrix[i][jj] -= coeff * a_matrix[j][jj]
            b[i] -= coeff*b[j]

    result = np.array([0.0]*n)
    for i in range(n):
        ii = n-1-i
        sum = 0
        result[ii] = b[ii]
        for j in range(ii+1, n):
            iter_count += 1
            result[ii] -= a_matrix[ii][j]*result[j]
        result[ii] /= a_matrix[ii][ii]

    return result


def lu_fact(a_matrix):
    global iter_count
    n = len(a_matrix)

    if n == 1:
        a0 = a_matrix[0][0]
        iter_count += 1
        return np.array([np.array([1.0])]), np.array([np.array([a0])])

    a_prev = []
    for i in range(n-1):
        row = []
        for j in range(n-1):
            row.append(a_matrix[i+1][j+1])
            iter_count += 1
        a_prev.append(np.array(row))
    a_prev = np.array(a_prev)
    for i in range(n-1):
        for j in range(n-1):
            a_prev[i][j] = a_prev[i][j] - a_matrix[0][j+1]*a_matrix[i+1][0]/a_matrix[0][0]

    lprev, uprev = lu_fact(a_prev)

    lnew = []
    unew = []
    for i in range(n):
        rowl = []
        rowu = []
        for j in range(n):
            iter_count += 1
            if (i!=0) and (j!=0):
                rowl.append(lprev[i-1][j-1])
                rowu.append(uprev[i-1][j-1])
            elif (i==0) and (j!=0):
                rowl.append(0)
                rowu.append(a_matrix[0][j])
            elif (i!=0) and (j==0):
                rowl.append(a_matrix[i][0]/a_matrix[0][0])
                rowu.append(0)
            else:
                rowl.append(1)
                rowu.append(a_matrix[0][0])
        lnew.append(np.array(rowl))
        unew.append(np.array(rowu))
    return np.array(lnew), np.array(unew)


def lu_method(a_matrix, b):
    global iter_count
    l, u = lu_fact(a_matrix)
    n = len(a_matrix)

    y = []

    for i in range(n):
        y.append(b[i])
        for j in range(i):
            iter_count += 1
            y[i] -= l[i][j] * y[j]
    y = np.array(y)

    x = []

    for i in range(n):
        iter_count += 1
        x.append(y[i])

    for i in range(n):
        ii = n-i-1
        for j in range(i):
            iter_count += 1
            jj = n-j-1
            x[ii] -= u[ii][jj]*x[jj]
        x[ii] /= u[ii][ii]

    return np.array(x)


def zeidel_method(a_matrix, b, eps):
    global iter_count
    n=len(b)
    xprev = np.array([1.0]*n)
    xcurr = np.array([1.0]*n)
    dist = 100
    while dist > eps:
        xcurr = np.copy(xprev)
        for i in range(n):
            sum = 0
            for j in range(n):
                iter_count += 1
                if j<i:
                    sum += a_matrix[i][j]*xcurr[j]
                elif j>i:
                    sum += a_matrix[i][j]*xprev[j]
            xcurr[i] = (b[i] - sum)/a_matrix[i][i]
        dist = distance(xcurr, xprev)
        xprev = xcurr
    return xcurr


def plot_cond_num_rand(n, k_max):
    cond_numbs = []
    for k in range(k_max + 1):
        avg = 0
        for j in range(1):
            avg += np.linalg.cond(gen_random(n, k))
        avg /= 100
        cond_numbs.append(avg)
    cond_numbs = np.array(cond_numbs)
    k_range = np.arange(0, k_max + 1, 1)
    plt.xlabel("K")
    plt.ylabel("Cond number")
    plt.plot(k_range, cond_numbs)
    plt.show()


def plot_cond_num_hilbert(n_max):
    cond_numbs = []
    for n in range(1, n_max + 1):
        cond_numbs.append(np.linalg.cond(gen_hilbert(n)))
    cond_numbs = np.array(cond_numbs)
    n_range = np.arange(1, n_max + 1, 1)
    plt.xlabel("N")
    plt.ylabel("Cond number")
    plt.plot(n_range, cond_numbs)
    plt.show()


def plot_test_gauss(n, k_max):
    b123 = []
    for i in range(n):
        b123.append(i + 1)
    b123 = np.array(b123)

    dist_arr = []
    for k in range(k_max + 1):
        avg = 0
        for j in range(5):
            generated = gen_random(n, k)
            avg += distance(gauss_method(generated, gen_b(generated)), b123)
        avg /= 5
        dist_arr.append(avg)
    k_range = np.arange(0, k_max + 1, 1)
    plt.xlabel("K")
    plt.ylabel("Avg error")
    plt.plot(k_range, dist_arr)
    plt.show()


def plot_test_gauss_to_cond(n, k_max):
    b123 = []
    for i in range(n):
        b123.append(i + 1)
    b123 = np.array(b123)

    dist_arr = []
    for k in range(k_max + 1):
        avg = 0
        for j in range(5):
            generated = gen_random(n, k)
            avg += distance(gauss_method(generated, gen_b(generated)), b123) / np.linalg.cond(generated)
        avg /= 5
        dist_arr.append(avg)
    k_range = np.arange(0, k_max + 1, 1)
    plt.xlabel("K")
    plt.ylabel("(Avg error)/(Cond)")
    plt.plot(k_range, dist_arr)
    plt.show()


def plot_zeidel_iterations_hilbert(n_max, eps):
    global iter_count
    iter_arr = []
    for n in range(1, n_max + 1):
        iter_count = 0
        generated = gen_hilbert(n)
        zeidel_method(generated, gen_b(generated), eps)
        iter_arr.append(iter_count)
    iter_arr = np.array(iter_arr)
    n_range = np.arange(1, n_max + 1, 1)
    plt.xlabel("N")
    plt.ylabel("Iterations")
    plt.plot(n_range, iter_arr)
    plt.show()


def plot_lu_iterations_hilbert(n_max):
    global iter_count
    iter_arr = []
    for n in range(1, n_max + 1):
        iter_count = 0
        generated = gen_hilbert(n)
        lu_method(generated, gen_b(generated))
        iter_arr.append(iter_count)
    iter_arr = np.array(iter_arr)
    n_range = np.arange(1, n_max + 1, 1)
    plt.xlabel("N")
    plt.ylabel("Iterations")
    plt.plot(n_range, iter_arr)
    plt.show()


# UNCOMMENT PLOT U WANT TO TEST

# COND NUMBER FOR RAND GEN MATRIXX
# plot_cond_num_rand(100, 200)

# COND NUMBER FOR HILBERT MATRIXX
# plot_cond_num_hilbert(20)

# GAUSS METHOD ERROR FOR RAND GEN MATRIXX
# plot_test_gauss(100, 10)

# GAUSS METHOD ERROR DIVIDED TO COND NUMBER
# plot_test_gauss_to_cond(100, 10)

# ZEIDEL METHOD ITERATIONS FOR HILBERT MATRIXX
# plot_zeidel_iterations_hilbert(50, 0.001)

# LU ITERATIONS FOR HILBERT MATRIXX
plot_lu_iterations_hilbert(50)


# TEST GAUSS FOR HILBERT MATRIXX, BAD RESULT FOR GAUSS IF N>10

iter_count = 0
n = 20
k = 0
generated = gen_hilbert(n)
print(gauss_method(generated, gen_b(generated)))