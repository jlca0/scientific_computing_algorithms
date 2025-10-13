from base import Solver
import time
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
import matplotlib.pyplot as plt

class QuadraticSolver:
    """Given the equation -nu * u'' - u^2 = fun it is numerically solved by different iterative methods. 
    The running time is measured and printed. The error of the stimation is also printed."""
    def __init__(self,x0, xf, t0, tf, nu, ua, ub, fun, left, right, exact = None):
        """ x0, xf (float) --- Interval where the ODE is solved.
            ua, ub (float) --- Boundary conditions.
            fun (callable) --- Function which defines the ODE.
            left (bool) --- If TRUE, the left boundary condition is Neumann. If FALSE it is Dirichlet. 
            right (bool) --- The same applies for the right boundary condition.
            exact (callable, optional) --- Exact solution of the problem. If given it automatically plots both the exact solution 
                      and the numerical solution"""
        super().__init__(x0, xf, nu, 0, 0, fun, left, right, exact)
    
    def solver_fixed_point(self, Internal_Call = False, N = None, tol = None, **kwargs):
        if N == None:
            raise ValueError("N (number of partitions) must be provided.")
        if tol == None:
            raise ValueError("tol (error tolerance) must be provided.")
        iter_max = 500  #Maximum number of iterations.
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        D = self.nu/dx2 * D
        D[0, 0] = 1.0
        D[0, 1] = 0.0
        D[N, N] = 1.0
        D[N, N - 1] = 0.0
        D = D.tocsc()
        #Resolución.
        u_old = np.zeros(N+1)
        u_new = np.zeros(N+1)
        error = tol + 1
        fun_x = self.fun(x)
        LU = splu(D)
        
        cont = 0 #Iteration counter.
        while(error >= tol and cont <500):
            b = fun_x + u_old**2
            b[0] = self.ua
            b[-1] = self.ub
            u_new = LU.solve(b)
            error = max(abs(u_new - u_old))
            u_old = u_new
            
            cont += 1
            
        if(cont == 500):
            print("Maximum number of iterations (", iter_max, ") was reached.")
            
        tf = time.time()  # We check the time.
        if not Internal_Call:
            print("Running time:", format(tf - t1))
            print("Iterations count:", cont)
            
        return x, u_new
    
    def solver_newton(self, Internal_Call = False, N = None, tol = None, **kwargs):
        #Initialization.
        if N == None:
            raise ValueError("N (number of partitions) must be provided.")
        if tol == None:
            raise ValueError("tol (error tolerance) must be provided.")
        iter_max = 500  #Maximum number of iterations.
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        U_diag = lil_matrix((N + 1, N + 1), dtype="float64")
        U_diag.setdiag(np.ones(N + 1), 0)
        U_diag = U_diag.tocsc()
        D = self.nu/dx2 * D
        D[0, 0] = 1.0
        D[0, 1] = 0.0
        D[N, N] = 1.0
        D[N, N - 1] = 0.0
        D = D.tocsc()
        
        #Resolution.
        u_old = np.zeros(N+1)
        u_new = np.zeros(N+1)
        error = tol + 1
        fun_x = self.fun(x)
                
        cont = 0 # Iteration counter.
        while(error >= tol and cont < iter_max):
            b = - self.nu/dx2 * D * u_old + u_old**2 + fun_x
            b[0] = - u_old[0] + ua
            b[-1] = - u_old[-1] + ub
            # Jacobian matrix.
            U_diag.setdiag(2 * u_old, 0)
            J = self.nu/dx2 * D - U_diag
            J[0, 0] = 1
            J[0, 1] = 0
            J[N, N] = 1
            J[N, N-1] = 0
            LU = splu(J)
            
            y_new = LU.solve(b)
            u_new = y_new + u_old
            error = max(abs(u_new - u_old))
            u_old = u_new
            
            cont += 1
            
        if(cont == 500):
            print("Maximum number of iterations (", iter_max, ") was reached.")
            
        tf = time.time()  # We check the time.
        if not Internal_Call:
            print("Running time:", format(tf - t1))
            print("Iterations count:", cont)
            
        return x, u_new
            