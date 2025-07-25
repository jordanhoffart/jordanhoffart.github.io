import triangle as tr
import scipy.sparse
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Solve - div (k grad u ) + qu = f on a 2D domain with
# u = g on the boundary of the domain

# Specify a mesh size
mesh_size = 0.1


# Pick some functions k, u, q, f, and g that satisfy the PDE in order to test
# your code
def u(x):
    return 0


def grad_u(x):
    return []


def k(x):
    return 0


def q(x):
    return 0


def g(x):
    return 0


# Define the vertices of your domain.
domain_vertices = [[]]

# Define the segments that connect the vertices of your domain.
segments = [[]]

# Implement a quadrature rule of the form
# integral_K f(x) dx = area(K) sum_n w_n f(x_n), where
# K is the reference triangle with vertices [0,0], [1,0], and [0,1].
# You can find tables of quadrature points and weights for the reference
# triangle online.
quadrature_points = [[]]
quadrature_weights = []
number_quadrature_points = len(quadrature_points)


# Define the basis functions on the reference triangle as well as their
# gradients.
def basis0(x):
    return 0


def basis1(x):
    return 0


def basis2(x):
    return 0


bases = [basis0, basis1, basis2]

grad_basis0 = [0, 0]
grad_basis1 = [0, 0]
grad_basis2 = [0, 0]
grad_bases = [grad_basis0, grad_basis1, grad_basis2]

# Let's evaluate our basis functions at the quadrature points. We will need these later.
bases_quad_points = [[phi(x) for x in quadrature_points] for phi in bases]

# Let's create our meshes.
tri = {"vertices": domain_vertices, "segments": segments}
mesh = tr.triangulate(tri, "pqa{}".format(mesh_size**2))

# Let's plot the first mesh.
plt.figure()
ax = plt.axes()
tr.plot(ax, **mesh)

# The elements are represented by a list of indices. The indices refer to
# which vertices in the vertices list make up the three vertices of our
# triangular element.
elements = mesh["triangles"]
vertices = mesh["vertices"]

# We have 1 degree of freedom per vertex.
number_vertices = len(vertices)
global_matrix = scipy.sparse.lil_matrix((number_vertices, number_vertices))
load_vector = np.zeros(number_vertices)

# Now let's assemble our system.
for element in elements:
    # Let's get the local element vertices.
    v0 = vertices[element[0]]
    v1 = vertices[element[1]]
    v2 = vertices[element[2]]

    # Let's compute all the relevant affine map information needed for our calculations.
    B = np.array([0, 0]).transpose()
    detB = abs(np.linalg.det(B))
    invB = np.linalg.inv(B)

    # Now let's compute the local entries and add them to the global system.
    for i in range(0, 3):
        row = element[i]

        # Compute the local load vector contribution via quadrature.
        for n in range(0, number_quadrature_points):
            xn = quadrature_points[n]
            wn = quadrature_weights[n]

            # Compute the local quadrature point by mapping it from the
            # reference element
            Txn = 0

            phi_in = bases_quad_points[i][n]
            # Compute the local contribution to the load vector:
            load_vector[row] += 0

        for j in range(0, 3):
            col = element[j]

            # Compute the local matrix contribution via quadrature.
            for n in range(0, number_quadrature_points):
                xn = quadrature_points[n]
                wn = quadrature_weights[n]

                # Compute the local quadrature point
                Txn = 0

                # Compute the stiffness term
                grad_phi_i = np.matmul(grad_bases[i], invB)
                grad_phi_j = np.matmul(grad_bases[j], invB)
                dot_ij = np.dot(grad_phi_i, grad_phi_j)
                stiff = 0

                # Compute the mass term
                phi_in = bases_quad_points[i][n]
                phi_jn = bases_quad_points[j][n]
                mass = 0

                global_matrix[row, col] += 0

# Now let's apply the Dirichlet boundary conditions. Each boundary edge is represented by a pair of indices.
# The indices tell us which vertices of the mesh are the endpoints of our edge.
boundary_edges = mesh["segments"]
for edge in boundary_edges:
    for i in edge:
        for j in range(0, number_vertices):
            continue
        continue

# Solve
uh = scipy.sparse.linalg.spsolve(global_matrix.tocsr(), load_vector)

# Compute the L2 error on the solution and the L2 error on the gradient of the solution via quadrature.
L2error = 0
H1semi_error = 0
for element in elements:
    # Get local vertices
    v0 = vertices[element[0]]
    v1 = vertices[element[1]]
    v2 = vertices[element[2]]

    # Get coefficients of numerical solution on the local element
    uh0 = uh[element[0]]
    uh1 = uh[element[1]]
    uh2 = uh[element[2]]

    # Compute relevant affine map information
    B = np.array([0, 0]).transpose()
    detB = abs(np.linalg.det(B))
    invB = np.linalg.inv(B)

    # Compute gradient of numerical solution on local element
    grad_phi_0 = np.matmul(grad_bases[0], invB)
    grad_phi_1 = np.matmul(grad_bases[1], invB)
    grad_phi_2 = np.matmul(grad_bases[2], invB)
    grad_uh = 0

    for n in range(0, number_quadrature_points):
        xn = quadrature_points[n]
        wn = quadrature_weights[n]

        # Compute the local quadrature point
        Txn = 0

        # Compute the exact solution at the quadrature point
        un = u(Txn)
        grad_un = grad_u(Txn)

        # Compute the H1 semi error
        dot_diff = np.dot(grad_uh - grad_un, grad_uh - grad_un)
        # We divide by 2 because that's the area of the reference triangle
        H1semi_error += 0

        # Compute the numerical solution at the quadrature point
        phi_0n = bases_quad_points[0][n]
        phi_1n = bases_quad_points[1][n]
        phi_2n = bases_quad_points[2][n]
        uhn = 0
        # We divide by 2 because that's the area of the reference triangle
        L2error += 0

# Let's display the error to the screen
print("h:", mesh_size, "L2:", np.sqrt(L2error), "H1:", np.sqrt(H1semi_error))

# Let's plot the numerical solution on the most refined mesh
plt.figure()
ax = plt.axes(projection="3d")
xs = [v[0] for v in vertices]
ys = [v[1] for v in vertices]
ax.plot_trisurf(xs, ys, uh, triangles=elements, cmap="inferno")

plt.show()
