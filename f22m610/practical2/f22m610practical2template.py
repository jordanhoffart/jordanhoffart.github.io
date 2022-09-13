import scipy.sparse as ss 
import scipy.sparse.linalg as ssl
import numpy as np
import os
import matplotlib.pyplot as plt
import tabulate as tb

def make_system_problem_1(A, b, num_x, num_y, I, h):
    for i in range(num_x):
        for j in range(num_y):
            if i in [0,num_x-1] or j in [0,num_y-1]:
                return # replace this
            else:
                return # replace this

def define_manufactured_solution():
    um = lambda x, y : 0 # edit this
    f = lambda x, y : 0 # edit this
    return um, f

def make_system_problem_2(A, b, num_x, num_y, I, h, um, f):
    for i in range(num_x):
        for j in range(num_y):
            if i in [0,num_x-1] or j in [0,num_y-1]:
                return # replace this
            else:
                return # replace this

def problem1():
    error_table = []
    for k in [1,2,3,4,5,6,7]:
        h = 1/2**k
        num_x = 2**k + 1
        num_y = 2**k + 1
        num_xy = num_x*num_y
        x = [i*h for i in range(num_x)]
        y = [j*h for j in range(num_y)]
        A = ss.lil_matrix((num_xy,num_xy))
        b = np.zeros(num_xy)
        I = np.zeros((num_x, num_y),dtype = np.int32)
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                I[i,j] = i_xy
                i_xy += 1
        make_system_problem_1(A, b, num_x, num_y, I, h)
        w = ssl.spsolve(A.tocsr(),b)
        v = np.zeros((num_x,num_y))
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                v[i,j] = w[i_xy]
                i_xy += 1
        if not os.path.exists('f22m610practical2/problem1'):
            os.makedirs('f22m610practical2/problem1')
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('p1k{}u_h'.format(k))
        ax.plot_surface(xx, yy, v, cmap='rainbow')
        fig.savefig('f22m610practical2/problem1/p1k{}u_h.png'.format(k))
        plt.close()
        v0 = 0
        for i in range(num_x):
            for j in range(num_y):
                if x[i] == 0.5 and y[j] == 0.5:
                    v0 = v[i,j]
        error_table.append([k, h, v0])
    for i in [0,1,2,3,4,5,6]:
        if i > 1:
            v00 = error_table[i][2]
            v01 = error_table[i-1][2]
            v02 = error_table[i-2][2]
            p = np.log2((v01-v02)/(v00-v01))
            error_table[i].append(p)
        else:
            error_table[i].append(0)
    fl = open('f22m610practical2/problem1/error_table_1_plain.txt','w')
    fl.write(tb.tabulate(error_table, headers=['h','k','u_h(0.5,0.5)','order']))
    fl.close()
    fl = open('f22m610practical2/problem1/error_table_1_latex.txt','w')
    fl.write(tb.tabulate(error_table, headers=['h','k','u_h(0.5,0.5)','order'],tablefmt='latex'))
    fl.close()

def problem2(um, f):
    error_table = []
    for k in [1,2,3,4,5,6,7]:
        h = 1/2**k
        num_x = 2**k + 1
        num_y = 2**k + 1
        num_xy = num_x*num_y
        x = [i*h for i in range(num_x)]
        y = [j*h for j in range(num_y)]
        A = ss.lil_matrix((num_xy,num_xy))
        b = np.zeros(num_xy)
        I = np.zeros((num_x, num_y),dtype = np.int32)
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                I[i,j] = i_xy
                i_xy += 1
        make_system_problem_2(A, b, num_x, num_y, I, h, um, f)
        w = ssl.spsolve(A.tocsr(),b)
        v = np.zeros((num_x,num_y))
        i_xy = 0
        for i in range(num_x):
            for j in range(num_y):
                v[i,j] = w[i_xy]
                i_xy += 1
        if not os.path.exists('f22m610practical2/problem2'):
            os.makedirs('f22m610practical2/problem2')
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('p2k{}u_h'.format(k))
        ax.plot_surface(xx, yy, v, cmap='rainbow')
        fig.savefig('f22m610practical2/problem2/p2k{}u_h.png'.format(k))
        plt.close()
        eh = np.zeros((num_x, num_y))
        for i in range(num_x):
            for j in range(num_y):
                eh[i,j] = v[i,j] - um(i*h,j*h)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('p2k{}e_h'.format(k))
        ax.plot_surface(xx, yy, eh, cmap='rainbow')
        fig.savefig('f22m610practical2/problem2/p2k{}e_h.png'.format(k))
        plt.close()
        l2_err = 0
        for i in range(num_x):
            for j in range(num_y):
                l2_err += eh[i,j]**2*h**2
        l2_err = np.sqrt(l2_err)
        linf_err = 0
        for i in range(num_x):
            for j in range(num_y):
                linf_err = max(linf_err, abs(eh[i,j]))
        error_table.append([k, h, l2_err, linf_err])
    for i in [0,1,2,3,4,5,6]:
        if i > 1:
            l2_0 = error_table[i][2]
            l2_1 = error_table[i-1][2]
            l2_2 = error_table[i-2][2]
            linf0 = error_table[i][3]
            linf1 = error_table[i-1][3]
            linf2 = error_table[i-2][3]
            p2 = np.log2((l2_1-l2_2)/(l2_0-l2_1))
            pinf = np.log2((linf1 - linf2)/(linf0 - linf1))
            error_table[i].append(p2)
            error_table[i].append(pinf)
        else:
            error_table[i].append(0)
            error_table[i].append(0)
    fl = open('f22m610practical2/problem2/error_table_2_plain.txt','w')
    fl.write(tb.tabulate(error_table, headers=['h','k','L2 error','Linf error', 'L2 order', 'Linf order']))
    fl.close()
    fl = open('f22m610practical2/problem2/error_table_2_latex.txt','w')
    fl.write(tb.tabulate(error_table, headers=['h','k','L2 error','Linf error', 'L2 order', 'Linf order'],tablefmt='latex'))
    fl.close()

def main():
    problem1()
    um, f = define_manufactured_solution()
    problem2(um, f)
    
if __name__ == '__main__':
    main()


