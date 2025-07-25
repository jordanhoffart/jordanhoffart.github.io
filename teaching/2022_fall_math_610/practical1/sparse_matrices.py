import scipy.sparse
import numpy
import scipy.sparse.linalg

m = 3
n = 3
sparse_matrix = scipy.sparse.lil_matrix((m,n))

sparse_matrix[0,0] = 1
sparse_matrix[0,1] = 1
sparse_matrix[1,1] = 2
sparse_matrix[2,1] = 1
sparse_matrix[2,2] = 3
print('sparse_matrix =\n',sparse_matrix.toarray())

sparse_matrix = sparse_matrix.tocsr()

b = numpy.array([0,1,1])
x = scipy.sparse.linalg.spsolve(sparse_matrix, b)
print('x =',x)