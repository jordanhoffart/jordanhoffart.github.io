import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import tabulate as tb

# problem data / exact solution
p0 = 2
r = 1


def f(x):
    return (1 + 1 / 8 / np.pi**2) * np.sin(2 * np.pi * x)


def u(x):
    return 1 / 8 / np.pi**2 * np.sin(2 * np.pi * x)


# reference transformation
def T(v0, h, s):
    return v0 + h * s


# quadrature info
w = [1, 1]
q = [-1 / np.sqrt(3), 1 / np.sqrt(3)]


def quad(f, a, b):
    return (b - a) / 2 * w[0] * f((b - a) / 2 * q[0] + (b + a) / 2) + (b - a) / 2 * w[
        1
    ] * f((b - a) / 2 * q[1] + (b + a) / 2)


# main loop
table = []
errors = []
for k in [2, 3, 4, 5, 6, 7]:
    # mesh info
    N = 2**k
    h = 1 / N
    vertices = [i * h for i in range(0, N + 1)]
    elements = [[i, i + 1] for i in range(0, N)]

    # assemble system
    A = ss.lil_matrix((len(vertices), len(vertices)))
    F = np.zeros(len(vertices))
    for element in elements:
        v0 = vertices[element[0]]
        A[element[0], element[0]] += 0  # edit this
        A[element[0], element[1]] += 0  # edit this
        A[element[1], element[0]] += 0  # edit this
        A[element[1], element[1]] += 0  # edit this
        F[element[0]] += quad(lambda s: 0, 0, 1)  # edit this
        F[element[1]] += quad(lambda s: 0, 0, 1)  # edit this

    # postprocess boundary conditions
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0
    F[0] = 0
    F[-1] = 0

    # solve
    U = ssl.spsolve(A.tocsr(), F)

    # plot
    if k == 7:
        plt.plot(vertices, U)
        plt.savefig("plot.png")

    # compute / record error
    ux = [u(x) for x in vertices]
    error = np.linalg.norm(U - ux) * np.sqrt(h)
    errors.append(error)

    # compute order of convergence
    if k in [4, 5, 6, 7]:
        order = np.log2((errors[-2] - errors[-3]) / (errors[-1] - errors[-2]))
    else:
        order = " "

    # update table
    entry = [k, h, error, order]
    table.append(entry)

# save table
fl = open("error_table.txt", "w")
fl.write(tb.tabulate(table, headers=["k", "h", "error", "order"]))
fl.close()
