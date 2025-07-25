import scipy.sparse as ss
import numpy as np
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import tabulate as tb


def main():
    # define problem parameters and exact solution u
    a = 0
    b = 1
    rho = 1
    q = 64
    r = 0
    f = lambda x: 0
    g_a = 0
    g_b = 1
    u = lambda x: (1 - np.exp(q * x)) / (1 - np.exp(q))
    switch = "centered"

    # generate error table as well as most-refined approximation with nodes
    # stored in x7 and u_i values stored in U7
    error_table, x7, U7 = tabulate_errors(a, b, rho, q, r, f, g_a, g_b, u, switch)
    save_error_table(error_table)

    # make plots of most refined approximation and error
    make_plots(x7, U7, u)


def tabulate_errors(a, b, rho, q, r, f, g_a, g_b, u, switch):
    error_table = []
    for k in [1, 2, 3, 4, 5, 6, 7]:
        # compute vector of x_i stored in x, vector of u_i stored in U, and mesh size h
        x, U, h = compute_approximation(a, b, k, rho, q, r, f, g_a, g_b, switch)
        L2_error, Linf_error = compute_errors(x, U, u)
        error_table.append([k, h, L2_error, Linf_error])
    return error_table, x, U


def compute_approximation(a, b, k, rho, q, r, f, g_a, g_b, switch):
    N = 2**k
    h = (b - a) / 2**k
    x = [a + i * h for i in range(0, N + 1)]
    A = assemble_system_matrix(N, rho, q, r, h, switch)
    b = assemble_load_vector(N, g_a, g_b, f, x)
    U = solve_system(A, b)
    return x, U, h


def assemble_system_matrix(N, rho, q, r, h, switch):
    A = ss.lil_matrix((N + 1, N + 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    if switch == "centered":
        for i in range(1, N):
            A[i, i - 1] = 0  # edit this
            A[i, i] = 0  # edit this
            A[i, i + 1] = 0  # edit this
    else:
        for i in range(1, N):
            A[i, i - 1] = 0  # edit this
            A[i, i] = 0  # edit this
            A[i, i + 1] = 0  # edit this
    return A


def assemble_load_vector(N, g_a, g_b, f, x):
    b = np.zeros(N + 1)
    b[0] = g_a
    b[-1] = g_b
    for i in range(1, N):
        b[i] = 0  # edit this
    return b


def solve_system(A, b):
    return ssl.spsolve(A.tocsr(), b)


def compute_errors(x, U, u):
    L2_error = compute_L2_error(x, U, u)
    Linf_error = compute_Linf_error(x, U, u)
    return L2_error, Linf_error


def compute_L2_error(x, U, u):
    # compute L2 error by Gaussian quadrature on each subinterval [x_i, x_{i+1}]
    L2_error_sq = 0  # L2 error squared
    n = 2  # order of quadrature rule
    integrand = (
        lambda t: (np.interp(t, x, U) - u(t)) ** 2
    )  # u_h(t) = np.interp(t, x, U)
    for i in range(len(x) - 1):
        lower = x[i]
        upper = x[i + 1]
        L2_error_sq += gauss_quad(n, integrand, lower, upper)
    return np.sqrt(L2_error_sq)


def gauss_quad(n, integrand, lower, upper):
    # use n-th order gaussian quadrature to approximate an integral of a function integrand(x) over an interval [lower, upper]
    if n == 1:
        return (upper - lower) * integrand((upper + lower) / 2)
    if n == 2:
        x1 = (upper - lower) / 2 * (-1 / np.sqrt(3)) + (upper + lower) / 2
        x2 = (upper - lower) / 2 * (1 / np.sqrt(3)) + (upper + lower) / 2
        return (upper - lower) / 2 * (integrand(x1) + integrand(x2))


def compute_Linf_error(x, U, u):
    # approximate max{|u_h(x)-u(x)| : a <= x <= b} by evaluating |u_h(t_i) - u(t_i)| at lots of points
    # a <= t_1 < t_2 < ... < t_n <= b and then taking the largest
    Linf_error = 0
    for i in range(len(x) - 1):  # loop over each subinterval [x_i, x_{i+1}]
        t = np.linspace(x[i], x[i + 1], 3)  # sample points in the subinterval
        local_Linf_error = max([abs(np.interp(t_i, x, U) - u(t_i)) for t_i in t])
        Linf_error = max(Linf_error, local_Linf_error)
    return Linf_error


def save_error_table(error_table):
    fl = open("error_table.txt", "w")
    fl.write(tb.tabulate(error_table, headers=["k", "h", "L2 error", "Linf error"]))
    fl.close()


def make_plots(x, U, u):
    fig, ax = plt.subplots()
    ax.plot(x, U)
    ax.set_title("u_h")
    fig.savefig("u_h.png")

    fig, ax = plt.subplots()
    ax.plot(x, [U[i] - u(x[i]) for i in range(len(x))])
    ax.set_title("e_h")
    fig.savefig("e_h.png")


if __name__ == "__main__":
    main()
