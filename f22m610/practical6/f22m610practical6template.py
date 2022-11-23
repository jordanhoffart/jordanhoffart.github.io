import triangle as tr
import scipy.sparse as ss
import numpy as np
import scipy.sparse.linalg as ssl

def um(t,x): return np.exp(-t)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
def dum(t,x): return [np.pi*np.exp(-t)*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), np.pi*np.exp(-t)*np.cos(np.pi*x[1])*np.sin(np.pi*x[0])]
def f(t,x): return (2*np.pi**2-1)*um(t,x)

old_mesh = tr.triangulate({'vertices':[[0,0],[1,0],[1,1],[0,1]],'segments':[[0,1],[1,2],[2,3],[3,0]]},'pqa1')
phi = [lambda x : 1-x[0]-x[1], lambda x : x[0], lambda x : x[1]]
dphi = [[-1,-1],[1,0],[0,1]]
xq = [[0.5,0],[0.5,0.5],[0,0.5]]
wq = [1/6,1/6,1/6]
nq = len(xq)

T = 1
nt = 100
dt = T/nt

print('theta, k, h, l2, h1')

theta = 1

for kh in [1,2,3,4]:
    h = 1/2**kh
    a = h**2
    opts = 'pqra{}'.format(a)
    mesh = tr.triangulate(old_mesh,opts)
    old_mesh = mesh

    vs = mesh['vertices']
    nv = len(vs)
    es = mesh['triangles']
    bms = mesh['vertex_markers']
    
    A = ss.lil_matrix((nv,nv))
    M = ss.lil_matrix((nv,nv))
    F0 = np.zeros(nv)
    F1 = np.zeros(nv)
    U0 = np.zeros(nv)
    U1 = np.array([um(0,x) for x in vs])
    
    for e in es:
        v0 = vs[e[0]]
        v1 = vs[e[1]]
        v2 = vs[e[2]]
        B = np.array([v1-v0,v2-v0]).transpose()
        invB = np.linalg.inv(B)
        detB = abs(np.linalg.det(B))
        for ir in [0,1,2]:
            for ic in [0,1,2]:
                A[e[ir],e[ic]] += 0 # change this
                for iq in range(nq):
                    M[e[ir],e[ic]] += 0 # change this
            for iq in range(nq):
                F1[e[ir]] += wq[iq]*f(0,v0+np.matmul(B,xq[iq]))*phi[ir](xq[iq])*detB
    
    M = M.tocsr()
    A = A.tocsr()
    M0 = M + dt*(theta-1)*A
    M1 = M + dt*theta*A
    M1 = M1.tolil()
    b = np.zeros(nv)
    
    for it in range(nt):
        F0[:] = F1[:]
        U0[:] = U1[:]
        F1[:] = 0
        
        for e in es:
            v0 = vs[e[0]]
            v1 = vs[e[1]]
            v2 = vs[e[2]]
            B = np.array([v1-v0,v2-v0]).transpose()
            invB = np.linalg.inv(B)
            detB = abs(np.linalg.det(B))
            for ir in [0,1,2]:
                for iq in range(nq):
                    F1[e[ir]] += 0 # change this
        
        b[:] = M0.dot(U0) + dt*theta*F1 + dt*(1-theta)*F0

        for im, bm in enumerate(bms):
            if bm[0] == 1:
                M1[im,:] = 0
                M1[im,im] = 1
                b[im] = 0
        
        U1[:] = ssl.spsolve(M1.tocsr(), b)

    l2 = 0
    h1 = 0
    for e in es:
        v0 = vs[e[0]]
        v1 = vs[e[1]]
        v2 = vs[e[2]]
        B = np.array([v1-v0,v2-v0]).transpose()
        invB = np.linalg.inv(B)
        detB = abs(np.linalg.det(B))
        for iq in range(nq):
            l2 += wq[iq]*detB*(um(T,v0+np.matmul(B,xq[iq]))-U1[e[0]]*phi[0](xq[iq])-U1[e[1]]*phi[1](xq[iq])-U1[e[2]]*phi[2](xq[iq]))**2
            h1 += wq[iq]*detB*np.linalg.norm(dum(T,v0+np.matmul(B,xq[iq]))-U1[e[0]]*np.matmul(dphi[0],invB)-U1[e[1]]*np.matmul(dphi[1],invB)-U1[e[2]]*np.matmul(dphi[2],invB))**2
    l2 = np.sqrt(l2)
    h1 = np.sqrt(h1)

    print(theta, kh, h, l2, h1)

