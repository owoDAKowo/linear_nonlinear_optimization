import numpy as np
from matplotlib import pyplot as plt


def function(x):
    return x[0] * x[0] + 2 * x[1] * x[1] - 2 * x[0] * x[1] + 8 * x[0] - 18 * x[1]


def result(x, iter):
    print('\nРезультаты')
    print(f'x* = {x}')
    print(f'Количество итераций: {iter}')
    print(f'f(x{iter}) = {x[0] * x[0] + 2 * x[1] * x[1] - 2 * x[0] * x[1] + 8 * x[0] - 18 * x[1]}')


def df(x):
    dfdx0 = 2 * x[0] - 2 * x[1] + 8
    dfdx1 = 4 * x[1] - 2 * x[0] - 18
    return np.array((dfdx0, dfdx1))


def get_lambda(x, s):
    up = 2 * s[0] * x[0] + 4 * s[1] * x[1] - 2 * s[1] * x[0] - 2 * s[0] * x[1] + 8 * s[0] - 18 * s[1]
    down = 2 * s[0] ** 2 + 4 * s[1] * s[1] - 4 * s[0] * s[1]
    l = -up / down
    return l


def gradient_descent(x):
    iter = 0
    func = []
    while any(df(x) != 0):
        iter += 1
        s = -df(x)
        l = get_lambda(x, s)
        x = x + l * s
        func.append(function(x))
    result(x, iter)
    return x, func, iter


if __name__ == '__main__':
    for i in range(1):
        x0 = [-100, -143]
        print(f'Точка начального приближения: {x0}')
        x, func, iter = gradient_descent(x0)

        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        plt.plot(np.arange(iter), func)
        plt.scatter(np.arange(iter), func, color='red', s=20, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('f(x)')
        plt.title(f'$x_0$ = {x0[0], x0[1]}')
        plt.show()
