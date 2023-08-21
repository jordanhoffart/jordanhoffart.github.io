import triangle as tr
import matplotlib.pyplot as plt

k = 6  # change this to see different mesh refinements; if k > 6, then something weird happens
N = 2**k
h = 1 / N
a = h**2

domain_vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
domain_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
tri = {"vertices": domain_vertices, "segments": domain_edges}
opts = "pqa{}".format(a)

mesh = tr.triangulate(tri, opts)
fig, ax = plt.subplots()
tr.plot(ax, **mesh)
plt.show()
