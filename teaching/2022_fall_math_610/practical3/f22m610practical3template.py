import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import numpy as np
import os
import matplotlib.pyplot as plt
import tabulate as tb


def define_manufactured_solution():
    um = lambda x, y: 0  # edit this
    f = lambda x, y: 0  # edit this
    return um, f


def make_system_problem_1(A, b, num_x, num_y, I, h, um, f):
    for i in range(num_x):
        for j in range(num_y):
            if i in [0, num_x - 1] or j in [0, num_y - 1]:
                return  # edit this
            else:
                return  # edit this


def make_system_problem_2(A, b, num_x, num_y, I, h, um, f):
    for i in range(num_x):
        for j in range(num_y):
            if i in [0, num_x - 1] or j in [0, num_y - 1]:
                return  # edit this
            else:
                return  # edit this


def make_system_problem(num, A, b, num_x, num_y, I, h, um, f):
    if num == 1:
        make_system_problem_1(A, b, num_x, num_y, I, h, um, f)
    if num == 2:
        make_system_problem_2(A, b, num_x, num_y, I, h, um, f)


def problem(num, um, f):
    root = "f22m610practical3"
    prob_num = str(num)
    error_table = []
    for k in [1, 2, 3, 4, 5, 6, 7]:
        h = 1 / 2**k
        num_x = 2**k + 1
        num_y = 2**k + 1
        num_xy = num_x * num_y
        x = [i * h for i in range(num_x)]
        y = [j * h for j in range(num_y)]
        A = ss.lil_matrix((num_xy, num_xy))
        b = np.zeros(num_xy)
        I = np.zeros((num_x, num_y), dtype=np.int32)
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                I[i, j] = i_xy
                i_xy += 1
        make_system_problem(num, A, b, num_x, num_y, I, h, um, f)
        w = ssl.spsolve(A.tocsr(), b)
        v = np.zeros((num_x, num_y))
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                v[i, j] = w[i_xy]
                i_xy += 1
        if not os.path.exists(root + "/problem" + prob_num):
            os.makedirs(root + "/problem" + prob_num)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_title("p{}k{}u_h".format(prob_num, k))
        ax.plot_surface(xx, yy, v, cmap="rainbow")
        fig.savefig(root + "/problem{}/p{}k{}u_h.png".format(prob_num, prob_num, k))
        plt.close()
        eh = np.zeros((num_x, num_y))
        for i in range(num_x):
            for j in range(num_y):
                eh[i, j] = v[i, j] - um(i * h, j * h)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_title("p{}k{}e_h".format(prob_num, k))
        ax.plot_surface(xx, yy, eh, cmap="rainbow")
        fig.savefig(root + "/problem{}/p{}k{}e_h.png".format(prob_num, prob_num, k))
        plt.close()
        l2_err = 0
        for i in range(num_x):
            for j in range(num_y):
                l2_err += eh[i, j] ** 2 * h**2
        l2_err = np.sqrt(l2_err)
        linf_err = 0
        for i in range(num_x):
            for j in range(num_y):
                linf_err = max(linf_err, abs(eh[i, j]))
        error_table.append([k, h, l2_err, linf_err])
    for i in [0, 1, 2, 3, 4, 5, 6]:
        if i > 1:
            l2_0 = error_table[i][2]
            l2_1 = error_table[i - 1][2]
            l2_2 = error_table[i - 2][2]
            linf0 = error_table[i][3]
            linf1 = error_table[i - 1][3]
            linf2 = error_table[i - 2][3]
            p2 = np.log2((l2_1 - l2_2) / (l2_0 - l2_1))
            pinf = np.log2((linf1 - linf2) / (linf0 - linf1))
            error_table[i].append(p2)
            error_table[i].append(pinf)
        else:
            error_table[i].append(0)
            error_table[i].append(0)
    fl = open(
        root + "/problem{}/error_table_{}_plain.txt".format(prob_num, prob_num), "w"
    )
    fl.write(
        tb.tabulate(
            error_table,
            headers=["h", "k", "L2 error", "Linf error", "L2 order", "Linf order"],
        )
    )
    fl.close()
    fl = open(
        root + "/problem{}/error_table_{}_latex.txt".format(prob_num, prob_num), "w"
    )
    fl.write(
        tb.tabulate(
            error_table,
            headers=["h", "k", "L2 error", "Linf error", "L2 order", "Linf order"],
            tablefmt="latex",
        )
    )
    fl.close()


def main():
    um, f = define_manufactured_solution()
    problem(1, um, f)
    problem(2, um, f)


if __name__ == "__main__":
    main()
