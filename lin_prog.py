from Simplex import *


def write_pickle(c, a, b, bounds, limits, filepath='data.pickle'):
    data = {
        'c': c,
        'a': a,
        'b': b,
        'bounds': bounds,
        'limits': limits
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


c = np.array([20, 13])  # коэф-ты целевой функции

a = np.array([[4, 3],  # коэф-ты левой части
              [3, 4],
              [4, 2],
              [4, 3]])

limits = [False, False, False, False]

b = np.array([20, 20, 30, 20])  # коэф-ты правой части

bounds = np.array([[0, 59],
                   [0, 42]])

lin = SimplexMethod('data.pickle')

fun, x = lin.optimal_solution(True, False)

# print(basis)

lin.experient([0, 500])
