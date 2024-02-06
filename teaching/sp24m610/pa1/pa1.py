import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def generate_mesh(n,a,b):
    """
        Return a list of n+1 evenly spaced points on the interval (a,b)
    """
    h = (b-a)/n
    return [a + i * h for i in range(n+1)]

def compute_num_dofs(degree,n):
    """
        Compute the number of global dofs when using the continuous Lagrange finite element space of a certain degree with n elements
    """
    if degree != 1:
        print("Error: we have only implemented the degree 1 case")
        return False
    else:
        return n-1

def get_elements(mesh):
    """
        Return a list of the subintervals (x_j,x_j+1) from the mesh
    """
    elements = []
    for j in range(len(mesh)-1):
        element = [mesh[j],mesh[j+1]]
        elements.append(element)
    return elements

def get_local_indices_interior():
    """
        Returns the local indices [0,1,...,n] where n is the number of local shape functions on an interior element.
    """
    # only implement the linear case
    return [0,1]

def get_local_indices_boundary():
    """
        Returns the local indices [0,1,...,n] where n is the number of local shape functions on a boundary element.
    """
    # only implement the linear case
    return [0]

def get_global_indices(k,num_elements):
    """
        Returns the global indices of the shape functions on the kth element.
    """
    if k > num_elements - 1:
        print("k is too big")
        return False 

    if k < 0:
        print("k is too small")
        return False

    # only implement the linear case with dirichlet bc
    if k == 0:
        return [0]
    elif k == num_elements-1:
        return [k-1]
    else:
        return [k-1,k]

def local_shape_function(degree,i,x):
    """ 
        Evaluate the ith local shape function of a certain degree at the point x in (0,1)
    """
    if degree != 1:
        print("Error: we have only implemented the degree 1 case")
        return False

    # only do linear case 
    if i == 0:
        return 1-x
    if i == 1:
        return x

    print("Error, something went wrong here.")
    return False

def local_shape_derivative(degree,i,x):
    """ 
        Evaluate the derivative of the ith local shape function of a certain degree at the point x in (0,1)
    """
    if degree != 1:
        print("Error: we have only implemented the degree 1 case")
        return False

    # only do linear case 
    if i == 0:
        return -1
    if i == 1:
        return 1

    print("Error, something went wrong here.")
    return False
    
def get_reference_quad_points(degree):
    """
        Returns the quadrature points on the unit interval that gives exact quadrature for polynomials up to a certain degree
    """
    transform = lambda x : (x + 1)/2 # map (-1,1) to (0,1)
    if degree < 1:
        return [transform(0)]
    if degree < 3:
        return [transform(x) for x in [-1/np.sqrt(3),1/np.sqrt(3)]]
    if degree < 5:
        return [transform(x) for x in [-np.sqrt(3/5),0,np.sqrt(3/5)]]
    print("Not implemented for the degree requested")
    return False

def get_reference_quad_weights(degree):
    """
        Returns the quadrature weights on the unit interval that gives exact quadrature for polynomials up to a certain degree
    """
    transform = lambda w : w / 2 # map the weights in the right way
    if degree < 1:
        return [transform(2)]
    if degree < 3:
        return [transform(w) for w in [1,1]]
    if degree < 5:
        return [transform(w) for w in [5/9,8/9,5/9]]
    print("Not implemented for the degree requested")
    return False

def assemble_local_stiffness_matrix(element, local_stiffness_matrix, local_indices):
    for i in local_indices:
        for j in local_indices:
            local_stiffness_matrix[i,j] = compute_stiffness_integral(i,j,element)

def assemble_local_mass_matrix(element, local_mass_matrix, local_indices):
    for i in local_indices:
        for j in local_indices:
            local_mass_matrix[i,j] = compute_mass_integral(i,j,element)

def assemble_local_rhs(element, local_rhs, local_indices):
    for i in local_indices:
        local_rhs[i] = compute_rhs_integral(i,element)

def quadrature(integrand):
    """
    Integrates a function integrand(x) over the unit interval (0,1)
    """
    degree = 3
    quad_points = get_reference_quad_points(degree)
    weights = get_reference_quad_weights(degree)
    integral = 0
    if len(quad_points) != len(weights):
        print("Something bad is about to happen")
        return False
    for q in range(len(quad_points)):
        integral += integrand(quad_points[q]) * weights[q]
    return integral

def compute_stiffness_integral(i,j,element):
    """
        Computes the integral involving the stiffness matrix terms
    """
    grad_i = lambda x : local_shape_derivative(1,i,x)
    grad_j = lambda x : local_shape_derivative(1,j,x)
    h = element[1] - element[0]
    integrand = lambda x : grad_i(x) * grad_j(x) / h
    return quadrature(integrand) 

def compute_rhs_integral(i,element):
    """
        Computes the integral involving the rhs terms
    """
    shape_i = lambda x : local_shape_function(1,i,x)
    h = element[1] - element[0]
    q = 200
    D = 8.8e7
    l = 50
    transform = lambda x : l * x
    rhs = lambda x : q * x * (l - x) / 2 / D
    integrand = lambda x : rhs(transform(x)) * shape_i(x) * h 
    return quadrature(integrand) 

def compute_mass_integral(i,j,element):
    """
        Computes the integral involving the mass matrix terms
    """
    shape_i = lambda x : local_shape_function(1,i,x)
    shape_j = lambda x : local_shape_function(1,j,x)
    h = element[1] - element[0]
    integrand = lambda x : shape_i(x) * shape_j(x) * h 
    return quadrature(integrand) 

def add_to_global_matrix(global_matrix,local_matrix,global_indices):
    for i in range(len(local_matrix)):
        for j in range(len(local_matrix)):
            global_matrix[global_indices[i],global_indices[j]] += local_matrix[i,j]

def add_to_global_rhs(global_rhs,local_rhs,global_indices):
    for i in range(len(local_rhs)):
            global_rhs[global_indices[i]] += local_rhs[i]

def main():
    print("Programming Assignment 1")

    #a = np.array([[1,2],[3,5]])
    #b = np.array([1,2])
    #x = np.linalg.solve(a,b)
    #print(x)
    #print(np.allclose(np.dot(a,x),b))

    num_elements = [25,50,100,200]
    left = 0 # left endpoint 
    right = 50 # right endpoint 
    degree = 1 # degree of the finite element space

    for n in num_elements:
        print("number elements:",n)
        mesh = generate_mesh(n,left,right)
        print("number nodes:",len(mesh))

        num_dofs = compute_num_dofs(degree,n)
        print("number dofs:", num_dofs)

        stiffness_matrix = sp.lil_matrix((num_dofs,num_dofs))
        mass_matrix = sp.lil_matrix((num_dofs,num_dofs))
        rhs = np.zeros(num_dofs)
        
        elements = get_elements(mesh)
        print("number elements:",len(elements))
        print("first element:", elements[0])
        print("last element:",elements[-1])

        # assemble interior elements first 
        for k in range(1,len(elements)-1):
            local_indices = get_local_indices_interior()
            global_indices = get_global_indices(k, n)
            num_local_shape_functions = len(local_indices)
            local_stiffness_matrix = np.zeros((num_local_shape_functions,num_local_shape_functions))
            local_mass_matrix = np.zeros((num_local_shape_functions,num_local_shape_functions))
            local_rhs = np.zeros(num_local_shape_functions)

            assemble_local_stiffness_matrix(elements[k], local_stiffness_matrix, local_indices)
            assemble_local_mass_matrix(elements[k], local_mass_matrix, local_indices) 
            assemble_local_rhs(elements[k], local_rhs, local_indices)

            add_to_global_matrix(stiffness_matrix,local_stiffness_matrix,global_indices)
            add_to_global_matrix(mass_matrix,local_mass_matrix,global_indices)
            add_to_global_rhs(rhs,local_rhs,global_indices)

        # now assemble boundary elements
        for k in range(0,len(elements)):
            local_indices = get_local_indices_boundary()
            global_indices = get_global_indices(k,n)
            num_local_shape_functions = len(local_indices)
            local_stiffness_matrix = np.zeros((num_local_shape_functions,num_local_shape_functions))
            local_mass_matrix = np.zeros((num_local_shape_functions,num_local_shape_functions))
            local_rhs = np.zeros(num_local_shape_functions)

            assemble_local_stiffness_matrix(elements[k], local_stiffness_matrix, local_indices)
            assemble_local_mass_matrix(elements[k], local_mass_matrix, local_indices) 
            assemble_local_rhs(elements[k], local_rhs, local_indices)

            add_to_global_matrix(stiffness_matrix,local_stiffness_matrix,global_indices)
            add_to_global_matrix(mass_matrix,local_mass_matrix,global_indices)
            add_to_global_rhs(rhs,local_rhs,global_indices)
        
        S = 1e3
        D = 8.8e-7
        gooch = stiffness_matrix + S/D*mass_matrix
        yee = sp.linalg.spsolve(gooch,rhs)
        plt.plot(mesh[1:-1],yee)
        plt.show()


if __name__ == "__main__":
    main()
