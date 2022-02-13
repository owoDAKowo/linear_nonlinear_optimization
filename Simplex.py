import pickle
import itertools as it
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class SimplexMethod:
    def __init__(self, filepath='data.pickle'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.c = data['c']
            self.a = data['a']
            self.b = data['b']
            self.bounds = data['bounds']
            self.limits = data['limits']
        f.close()
        self.preprocess()

    @staticmethod
    def draw_table(table, c_basis, cj, basis, more_info=False):
        print(pd.DataFrame(np.around(table, 2), columns=[f'P{i}' for i in range(1, table.shape[1])] + ['P0']),
              '\n')
        if more_info:
            print(f'c_basis: {c_basis}')
            print(f'cj: {cj}')
            print(f'basis: {basis}')
            # print(f'p0:  {table.T[-1]}')
            print(f'delta: {np.around(np.dot(c_basis, table) - cj, 2)}')
            print('-' * 50)

    @staticmethod
    def minus1(tmp, n):
        t = []
        for i in tmp:
            if i > n:
                t.append(i - 1)
            else:
                t.append(i)
        return np.array(t)

    def teta(self, pi):
        p0 = self.table.T[-1]
        tmp = []
        for i in range(len(pi)):
            if pi[i] <= 0:
                tmp.append(100000)
                # print(p0[i])
            else:
                tmp.append(p0[i] / pi[i])
        return list(sorted(enumerate(tmp), key=lambda x: x[1]))

    def to_canon(self, tbl, limits):
        e = np.eye(tbl.shape[0])
        if len(limits) - sum(limits) == 0:
            return np.hstack((tbl, e))
        else:
            e[:len(limits) - sum(limits)] = e[:len(limits) - sum(limits)] * -1
        self.table = np.hstack((tbl, e))

    def add_artificial_basis(self, limits):
        for i in range(len(limits)):
            if not limits[i]:
                f = np.zeros(self.table.shape[0])
                f[i] = 1
                self.table = np.column_stack((self.table, f))

    def preprocess(self):
        if self.bounds[:, 1].all() != 0:
            a = np.concatenate((self.a, np.eye(self.bounds.shape[1])))
            b = np.concatenate((self.b, self.bounds[:, 1]))
            limits = self.limits
            for i in range(self.bounds.shape[1]):
                limits.append(True)

        self.to_canon(a, limits)
        self.add_artificial_basis(limits)
        self.table = np.concatenate((self.table, b.reshape(-1, 1)), axis=1)
        self.c_basis = np.hstack((np.ones(self.table.shape[0] - 2), np.zeros(2)))
        self.cj = np.hstack((np.hstack((np.zeros(8), np.ones(4))), [0]))
        self.basis = np.array([8, 9, 10, 11, 6, 7])

    def minimization(self, table, c_basis, cj, basis, draw_flag=False, info=False):
        iters = 0
        if draw_flag:
            self.draw_table(table, c_basis, cj, basis, info)
        while max((np.dot(c_basis, table) - cj)[:-1]) > 0:
            index = max(enumerate((np.dot(c_basis, table) - cj)[:-1]), key=lambda key: key[1])[0]
            ind = self.teta(table.T[index])
            c_basis[ind[0][0]] = cj[index]
            table[ind[0][0]] = table[ind[0][0]] / table[ind[0][0], index]
            for i in ind[1:]:
                table[i[0]] = table[i[0]] - table[ind[0][0]] * table[i[0], index]
            if basis[ind[0][0]] > 7:
                indx = basis[ind[0][0]]
                table = np.delete(table, indx, axis=1)
                cj = np.delete(cj, indx)
                basis = self.minus1(basis, basis[ind[0][0]])
            basis[ind[0][0]] = index
            if draw_flag:
                self.draw_table(table, c_basis, cj, basis, info)
            iters += 1
        return table, c_basis, cj, basis, iters

    def optimal_solution(self, draw_flag=False, info=False):
        self.table, self.c_basis, self.cj, self.basis, iters = self.minimization(self.table, self.c_basis, self.cj,
                                                                                 self.basis, draw_flag, info)
        self.cj[0] = self.c[0]
        self.cj[1] = self.c[1]
        self.c_basis = self.cj[self.basis]
        delta = np.dot(self.c_basis, self.table) - self.cj

        if max(delta[:-1]) > 0:
            self.table, self.c_basis, self.cj, self.basis, iters = self.minimization(self.table, self.c_basis, self.cj,
                                                                                     self.basis, draw_flag, info)

        x = np.zeros(len(self.c))
        x[self.basis[self.basis < len(self.c)]] = self.table.T[-1][self.basis < len(self.c)]
        self.x = x
        self.fun = delta[-1]
        print(f'x* = {self.x}\nf(x*) = {self.fun}\n')
        return self.fun, self.x

    def sens_analysis(self, c,flag=False):
        result = True
        cj = self.cj
        cj[0] = c[0]
        cj[1] = c[1]
        c_basis = cj[self.basis]
        delta = np.dot(c_basis, self.table) - cj

        if max(delta[:-1]) > 0:
            if flag:
                print(c, '\nfun = ', delta[-1])
                table, c_basis, cj, basis, iters = self.minimization(self.table.copy(), c_basis, cj, self.basis.copy())
            result = False
        if flag:
            table, basis = self.table, self.basis
            x = np.zeros(len(self.c))
            x[basis[basis < len(self.c)]] = table.T[-1][basis < len(self.c)]
            fun = (np.dot(c_basis, self.table) - cj)[-1]
            print(fun,'\n'+'-'*50)
        return result

    def experient(self,rng,c2=False):
        rng = np.arange(rng[0], rng[1]+1)
        tmp = []
        for i in it.product(rng, rng):
            if c2:
                i=(i[1],i[0])
            if self.sens_analysis(i, False):
                tmp.append(i)

        tmp = np.array(tmp).reshape(-1, 2)
        x, y = [], []
        for i in set(tmp[:, 0]):
            x.append(i)
            y.append([tmp[tmp[:, 0] == i][0, 1], tmp[tmp[:, 0] == i][-1, 1]])

        fig, ax = plt.subplots()
        ax.plot(x, y, 'r-')
        y = np.array(y).reshape(-1, 2)
        plt.fill_between(x, y[:, 0], y[:, 1], hatch='\\')
        if c2:
            plt.xlabel('c2')
            plt.ylabel('c1')
        else:
            plt.xlabel('c1')
            plt.ylabel('c2')
        plt.show()