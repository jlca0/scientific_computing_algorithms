import time
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
import matplotlib.pyplot as plt

class BVPSolver:
    """Given a boundary conditions problem of the form u - alpha * u'' = fun it is numerically solved by a finite
    difference scheme. The running time is measured and printed. The error of the stimation is also printed."""
    def __init__(self,x0, xf, alpha, ua, ub, fun, left, right, exact = None):
        """ x0, xf (float) --- Interval where the ODE is solved.
            ua, ub (float) --- Boundary conditions.
            fun (callable) --- Function which defines the ODE.
            left (bool) --- If TRUE, the left boundary condition is Neumann. If FALSE it is Dirichlet. 
            right (bool) --- The same applies for the right boundary condition.
            exact (callable, optional) --- Exact solution of the problem. If given it automatically plots both the exact solution 
                      and the numerical solution"""
        self.x0 = float(x0)
        self.xf = float(xf)
        self.alpha = float(alpha)
        self.ua = float(ua)
        self.ub = float(ub)
        self.left = bool(left)
        self.right = bool(right)
        self.fun = fun
        self.exact = exact
        self._validate_input()
    
    def solver(self, N, Internal_Call = False):
        """
          N (int) --- Number of partitions of the mesh."""
        t1 = time.time()
        dx, dx2, x, D, Id, b = self._mesh_and_sys_setting(N)
        if self.left and self.right:  # Both Neumann.
            # The first and last equations are changed.
            D[N, N - 1] = 2 * D[N, N - 1]
            D[0, 1] = 2 * D[0, 1]
        
            # The first and last entry of b are changed.
            b[0] -= self.ua * 2 * self.alpha / dx
            b[N] += self.ub * 2 * self.alpha / dx

        elif self.left:  # Left Neumann.
            D[N, N] = 0.0
            D[N, N - 1] = 0.0
            D[0, 1] = 2 * D[0, 1]

            b[0] -= self.ua * 2 * self.alpha / dx
            b[N] += self.ub

        elif self.right:  # Right Neumann.
            D[0, 0] = 0.0
            D[0, 1] = 0.0
            D[N, N - 1] = 2 * D[N, N - 1]

            b[0] += self.ua
            b[N] += self.ub * 2 * self.alpha / dx

        else:  # Both Dirichlet.
            D[0, 0] = 0.0
            D[0, 1] = 0.0
            D[N, N] = 0.0
            D[N, N - 1] = 0.0

            b[0] = self.ua
            b[N] = self.ub
        # Change from lil to csc for performance in matrix calculation.
        D = D.tocsc()
        # Notice that the matrix A is not symmetric.(Cholesky does not apply.)
        A = Id + self.alpha / dx2 * D
    
        # Solution of the system.
        # Complete LU factorization of the sparse matrix A. 
        LU = splu(A)
        usol = LU.solve(b)

        #We check the time and return the solution.
        if not Internal_Call:
            tf = time.time()
            print(f"Runnign time: {format(tf - t1)}")
        return x, usol
            
        
    def solver_penalization(self, N, Internal_Call = False):
        """
          N (int) --- Number of partitions of the mesh."""
        #Initialization and input data verification. 
        N = int(N)
        if(N < 2):
            raise ValueError("Choose bigger N.")
        M = 1e30  #Penalization factor
        t1 = time.time()
        dx, dx2, x, D, Id, b = self._mesh_and_sys_setting(N)
        
        # Clasification of the boundary conditions.
        if self.left:  # Left is Neumann.
            # Change the first and last row of the matrix D.
            D[0, 1] = 2 * D[0, 1]
            
            # Change the first and last entry of the vector b.
            b[0] -= self.ua * 2 * self.alpha / dx
        else: # Left is not Neumann.
            D[0, 0] = 0.0
            D[0, 1] = 0.0
            
            b[0] = self.ua
        if self.right:  # Right es Neumann.
            D[N, N - 1] = 2 * D[N, N - 1]

            b[N] += self.ub * 2 * self.alpha / dx
        else:  # Right is not Neumann.
            D[N, N] = 0.0
            D[N, N - 1] = 0.0
            
            b[N] = self.ub
        
        #We go from lil to csc for performance in matrix operation.
        D = D.tocsc()
        A = Id + self.alpha / dx2 * D  #Notice A is non-symmetric    

        # Solution of the system.
        # Complete LU factorization of the sparse matrix A, 
        # it could be Cholesky or Conjugate Gradient which suit symmetryc matrices.
        LU = splu(A)
        usol = LU.solve(b)

        #We check the time and return the solution.
        if not Internal_Call:
            tf = time.time()
            print(f"Runnign time: {format(tf - t1)}")
        return x, usol
        
    def solution_plot(self, N, penalization = False):
        if (self.exact != None):
            if(penalization):
                x, usol = self.solver_penalization(N,Internal_Call = True) 
                plt.plot(x, usol, "b", x, self.exact(x), "r")
            else:
                x, usol = self.solver(N,Internal_Call = True) 
                plt.plot(x, usol, "b", x, self.exact(x), "r")
        else:    
            raise ValueError("No exact solution was provided.")
    
    def solution_error(self, N, penalization = False):
        if (self.exact != None):
            if(penalization):
                x, usol = self.solver_penalization(N,Internal_Call = True) 
                err = max(abs(usol - self.exact(x)))
                print(f"Discretization error: {format(err)}")
                return err
            else:
                x, usol = self.solver(N,Internal_Call = True) 
                err = max(abs(usol - self.exact(x)))
                print(f"Discretization error: {format(err)}")
                return err
        else:    
            raise ValueError("No exact solution was provided.")
            
    def _validate_input(self):
        if(self.x0 >= self.xf):
            raise ValueError("Degenerate interval.")
        elif(self.alpha <= 0):
            raise ValueError("Alpha must be positive.")
            
    def _mesh_and_sys_setting(self, N):
        """Calculation of the mesh. Construction of D, before imposing boudary conditions. Contruction of the
        independent term b."""
        dx = (self.xf - self.x0) / N
        dx2 = dx * dx
        x = np.linspace(self.x0, self.xf, N + 1)
        D = lil_matrix((N + 1, N + 1), dtype="float64")   
        D.setdiag(2.0 * np.ones(N + 1), 0)
        D.setdiag(-1.0 * np.ones(N), 1)
        D.setdiag(-1.0 * np.ones(N), -1)
        
        Id = identity(N + 1, dtype="float64", format="csc")
        
        b = self.fun(x)
        return dx, dx2, x, D, Id, b
    
#Now we check the method with multiple calls which allow to empirically calculate its order.
def f(x):
    """Function which defines the ODE we want to solve."""
    y = 2.0 * np.sin(x)
    return y

def uexact(x):
    """The exact solution of the problem given the Dirichlet conditions u(0) = u(pi) = 0 and alpha = 1"""
    y = np.sin(x)
    return y

equation = BVPSolver(0,np.pi,1,0,-1,f,False,True,uexact)  #We call an object from BVPSolver

mesh = [50, 100, 200, 400]
error0 = 1

print("Direct Boundary.")
plt.figure("No Penalization: Numerical Solution (Blue) and Exact Solution (Red)")
for N in mesh:
    error1 = equation.solution_error(N,False)
    if N > 50:
        print(f"With N = {N} the order is = {np.log(error0 / error1)/np.log(2)} \n")
    else:
        print(f"With N = {50} the order is = --- \n")
    error0 = error1

print("\n")
print("Penalization Direct Boundary.")
plt.figure("Penalization: Numerical Solution (Blue) and Exact Solution (Red)")
for N in mesh:
    error1 = equation.solution_error(N,True)
    if N > 50:
        print(f"With N = {N} the order is = {np.log(error0 / error1)/np.log(2)} \n")
    else:
        print(f"With N = {50} the order is = --- \n")
    error0 = error1
    
#We can also plot the solution.

equation.solution_plot(400)
