import time
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
import matplotlib.pyplot as plt

class Solver:
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
        
    def solver(self,Internal_Call = False, **kwargs):
        return None, None
        
    def solution_plot(self, **kwargs):
        if (self.exact != None):
                x, usol = self.solver(Internal_Call = True, **kwargs) 
                plt.plot(x, usol, "b", x, self.exact(x), "r")
        else:    
            raise ValueError("No exact solution was provided.")
    
    def solution_error(self, **kwargs):
        if (self.exact != None):
            x, usol = self.solver(Internal_Call = True, **kwargs) 
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
        
        return dx, dx2, x, D, Id
    