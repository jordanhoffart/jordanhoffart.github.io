import numpy as np
import matplotlib.pyplot as plt

# problem parameters
q = 200
D = 8.8e7
l = 50
S = 100
f = lambda x: q / 2 / D * x * (l - x)

# write the exact solution yourself
W = lambda x: 0  # notice that the variable here is x, not t

# mesh info
N = 25
h = l / N
nodes = [j * h for j in range(N + 1)]
elements = [[nodes[j], nodes[j + 1]] for j in range(len(nodes) - 1)]

# linear system
A = np.zeros((N - 1, N - 1))
F = np.zeros(N - 1)

# quadrature info
quad_pts = [0, -np.sqrt(3 / 5), np.sqrt(3 / 5)]
quad_wts = [8 / 9, 5 / 9, 5 / 9]
integrate = lambda integrand: sum(
    [integrand(quad_pts[q]) * quad_wts[q] for q in range(len(quad_pts))]
)

# basis functions
basis = [lambda x: (1 - x) / 2, lambda x: (1 + x) / 2]
dbasis = [lambda x: -1 / 2, lambda x: 1 / 2]

# assembly
for k in range(1, len(elements) - 1):
    # local vectors and matrices
    Fk = np.zeros(len(basis))
    Ak = np.zeros((len(basis), len(basis)))

    # element info
    element = elements[k]
    x0 = element[0]
    x1 = element[1]
    refmap = lambda x: (x0 + x1) / 2 + (x1 - x0) / 2 * x
    dmap = lambda x: (x1 - x0) / 2

    # local assembly
    for i in range(len(basis)):
        integrand = lambda x: 0  # replace this
        Fk[i] = integrate(integrand)
        for j in range(len(basis)):
            integrand = lambda x: i + j  # replace this
            Ak[i, j] = integrate(integrand)

    # assemble to global system
    for i in range(len(basis)):
        F[k - 1 + i] += Fk[i]
        for j in range(len(basis)):
            A[k - 1 + i, k - 1 + j] += Ak[i, j]

# assemble the local entry for k = 0 yourself
integrand = lambda x: 0  # replace this
F[0] += integrate(integrand)  # replace this
A[0, 0] += integrate(integrand)  # replace this

# also assemble k = N
F[-1] += integrate(integrand)  # replace this
A[-1, -1] += integrate(integrand)  # replace this

# solve
Wh = np.linalg.solve(A, F)

# plot
plt.plot(nodes[1:-1], Wh)
plt.show()

# look at pyplots documentation for how to save a plot

# or you can save the points as a list of x,y pairs to use in some other plotting software
# make sure you change the filename so you don't overwrite
plot_nodes = nodes[1:-1]
with open("data.txt", "w") as f:
    for j in range(len(plot_nodes)):
        f.write(",".join([str(plot_nodes[j]), str(Wh[j])]) + "\n")

# compute errors
L2_sq = 0
for k in range(1, len(elements) - 1):
    # element info
    element = elements[k]
    x0 = element[0]
    x1 = element[1]
    refmap = lambda x: (x0 + x1) / 2 + (x1 - x0) / 2 * x
    dmap = lambda x: (x1 - x0) / 2

    W0 = Wh[k - 1]
    W1 = Wh[k]
    integrand = lambda x: (
        W(refmap(x)) - W0 * basis[0](x) - W1 * basis[1](x)
    ) ** 2 * dmap(x)
    L2_sq += integrate(integrand)

# do the k = 0 and k = N-1 case yourself

# implement the H1 error and Linfinity errors yourself
H1_sq = 0
Linf = 0

# prints a comma separated line of the errors which can be put into a spreadsheet to compute error rates and produce tables
print(",".join([str(N), str(h), str(np.sqrt(L2_sq)), str(np.sqrt(H1_sq)), str(Linf)]))
