#solve -(ku')' + qu = f, u(0) = 0, u(1) = 0
#test with k(x) = 1 + x, q(x) = 1, and u(x) = x*(1-x)

import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot
import math

#problem data
def k(x):
    return 1 + x 
def q(x):
    return 1
def f(x): 
    return 1 + 4*x + x*(1-x)

#mesh generation
number_of_elements = 200
number_of_nodes = number_of_elements + 1
mesh_size = 1/number_of_elements 

#shape functions and derivatives, assuming reference element is [0,1] and linear elements

shapes = [lambda s: 1-s, lambda s: s] # left shape function, right shape function

shape_derivatives = [lambda s: -1, lambda s: 1]

#quadrature rules and reference map
#5 point gauss quadrature, assuming that the integration is over [0,1]
number_of_quad_points = 5

quad_weights = [0.5688888888888889/2, 0.4786286704993665/2, 0.4786286704993665/2, 0.2369268850561891/2, 0.2369268850561891/2] #found online

quad_points = [1/2*x + 1/2 for x in [0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640]] #found online

#the reference map
def reference_map(i, s):
    return i*mesh_size + s*mesh_size  # maps s in [0,1] to x in [x_i, x_{i+1}]

#system assembly
#pre-allocate / pre-compute as much as we can for efficiency
shapes_at_quad_points = [[shapes[i](x) for x in quad_points] for i in range(0, 2)] #entry [i][j] is the value of the ith shape function at the jth quad point

shape_derivatives_at_quad_points = [[shape_derivatives[i](x) for x in quad_points] for i in range(0, 2)] # entry [i][j] is the value of the ith shape derivative at the jth quad point 

global_matrix = scipy.sparse.lil_matrix((number_of_nodes, number_of_nodes))
load_vector = [0]*number_of_nodes

#assemble global system by looping over elements
for i in range(0, number_of_elements):
    #use quadrature to add local entries to system matrix and load vector
    for j in range(0, number_of_quad_points):
        k_at_quad_point = k(reference_map(i, quad_points[j]))
        q_at_quad_point = q(reference_map(i, quad_points[j]))

        stiffness_component = 0 #EDIT THIS
        mass_component = 0 #EDIT THIS
        global_matrix[i,i] += stiffness_component + mass_component 

        stiffness_component = 0 #EDIT THIS
        mass_component = 0 #EDIT THIS
        global_matrix[i,i+1] += stiffness_component + mass_component 

        stiffness_component = 0 #EDIT THIS
        mass_component = 0 #EDIT THIS
        global_matrix[i+1,i] += stiffness_component + mass_component 

        stiffness_component = 0 #EDIT THIS
        mass_component = 0 #EDIT THIS
        global_matrix[i+1,i+1] += stiffness_component + mass_component

        f_at_quad_point = f(reference_map(i, quad_points[j]))

        load_vector[i] += mesh_size*quad_weights[j]*f_at_quad_point*shapes_at_quad_points[0][j]
        
        load_vector[i+1] += mesh_size*quad_weights[j]*f_at_quad_point*shapes_at_quad_points[1][j]

#apply dirichlet boundary conditions
#left boundary
global_matrix[0,0] = 1
global_matrix[0,1] = 0
load_vector[0] = 0
#right boundary
global_matrix[-1,-1] = 1
global_matrix[-1,-2] = 0
load_vector[-1] = 0

#solve system
numerical_solution = scipy.sparse.linalg.spsolve(global_matrix.tocsr(), load_vector)

#compute errors
#the exact solution and its derivative
def exact_solution(x):
    return x*(1-x)

def exact_solution_derivative(x):
    return 1 - 2*x 

#the errors
L2_error_squared = 0
h1_semi_error_squared = 0
Linf_error = 0

#need special points for the Linf error
number_of_Linf_points = 11

Linf_points = [i/(number_of_Linf_points - 1)  for i in range(0, number_of_Linf_points)] #points are in reference interval [0,1]

#pre-allocate / precompute things to be efficient
shapes_at_Linf_points = [[shapes[i](x) for x in Linf_points] for i in range(0, 2)]

#calculate errors locally and sum / max over all elements
for i in range(0, number_of_elements):
    #add / update local L2, h1, and Linf errors
    for j in range(0, number_of_quad_points):
        exact_solution_at_quad_point = exact_solution(reference_map(i,quad_points[j]))

        numerical_solution_at_quad_point = numerical_solution[i]*shapes_at_quad_points[0][j] + numerical_solution[i+1]*shapes_at_quad_points[1][j]

        L2_error_squared += mesh_size*quad_weights[j]*(exact_solution_at_quad_point - numerical_solution_at_quad_point)**2

        exact_solution_derivative_at_quad_point = exact_solution_derivative(reference_map(i, quad_points[j]))

        numerical_solution_derivative_at_quad_point = (numerical_solution[i]*shape_derivatives_at_quad_points[0][j] + numerical_solution[i+1]*shape_derivatives_at_quad_points[1][j])/mesh_size

        h1_semi_error_squared += mesh_size*quad_weights[j]*(exact_solution_derivative_at_quad_point - numerical_solution_derivative_at_quad_point)**2
    
    for j in range(0, number_of_Linf_points):
        exact_solution_at_Linf_point = exact_solution(reference_map(i, Linf_points[j])) 
        
        numerical_solution_at_Linf_point = numerical_solution[i]*shapes_at_Linf_points[0][j] + numerical_solution[i+1]*shapes_at_Linf_points[1][j]

        abs_error_at_Linf_point = abs(exact_solution_at_Linf_point - numerical_solution_at_Linf_point)

        Linf_error = max(Linf_error, abs_error_at_Linf_point)

#calculate and print out mesh size and errors
print(mesh_size, math.sqrt(L2_error_squared), math.sqrt(L2_error_squared + h1_semi_error_squared), Linf_error)


#plot solution
nodes = [i*mesh_size for i in range(0, number_of_nodes)]
matplotlib.pyplot.plot(nodes, numerical_solution)
matplotlib.pyplot.show()

