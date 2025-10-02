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
    
class BVPSolver(Solver):
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
        super().__init__(x0, xf, alpha, ua, ub, fun, left, right, exact)
    
    
    def solver(self, Internal_Call = False, N = None, **kwargs):
        """
          N (int) --- Number of partitions of the mesh."""
        if N == None:
            raise ValueError("N (number of partitions must be provided.")
        N = int(N)
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        b = self.fun(x)
        
        # Clasification
        if self.left:  # Left is Neumann
            # Change first row of D
            D[0, 1] = 2 * D[0, 1]
            
            # Change first and last entries of b
            b[0] -= self.ua * 2 * self.alpha / dx
        else: 
            D[0, 0] = 0.0
            D[0, 1] = 0.0
            
            b[0] = self.ua
        if self.right:  # Right is Neumann
            D[N, N - 1] = 2 * D[N, N - 1]

            b[N] += self.ub * 2 * self.alpha / dx
        else:  
            D[N, N] = 0.0
            D[N, N - 1] = 0.0
            
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
    
class HTLSolver(Solver):
    """Given the heat equation u_t - alpha * u_xx = fun it is numerically solved by the implicit lines method. The running time is measured and printed. The error of the stimation is also printed."""
    def __init__(self,x0, xf, t0, tf, alpha, u0:callable, ua:callable, ub:callable, fun, left, right, exact = None):
        """ x0, xf (float) --- Interval where the ODE is solved.
            ua, ub (float) --- Boundary conditions.
            fun (callable) --- Function which defines the ODE.
            left (bool) --- If TRUE, the left boundary condition is Neumann. If FALSE it is Dirichlet. 
            right (bool) --- The same applies for the right boundary condition.
            exact (callable, optional) --- Exact solution of the problem. If given it automatically plots both the exact solution 
                      and the numerical solution"""
        super().__init__(x0, xf, alpha, 0, 0, fun, left, right, exact)
        self.t0 = t0
        self.tf = tf
        self.u0 = u0
        self.ua = ua
        self.ub = ub
        
    def solver(self, Internal_Call = False, N = None, M = None, theta = None, **kwargs):
        if (theta == 1):
            return self._explicit_solver(Internal_Call = True,N = N , M = M, theta = theta)
        else:
            return self._implicit_theta_solver(Internal_Call = True,N = N , M = M, theta = theta)
    
    def solution_plot(self, N = None, M = None, theta = None, **kwargs):
        if (self.exact != None):
                t, x, usol = self.solver(Internal_Call = True, N = N, M = M, theta = theta, **kwargs) 
                plt.plot(x, usol, "b", x, self.exact(x,t[-1]), "r")  # We graph the last time iteration.
        else:    
            raise ValueError("No exact solution was provided.")
    
    def solution_error(self, N = None, M = None, theta = None, **kwargs):
        if (self.exact != None):
            t, x, usol = self.solver(Internal_Call = True, N = N, M = M, theta = theta, **kwargs) 
            err = max(abs(usol - self.exact(x,t[-1])))
            print(f"Discretization error: {format(err)}")
            return err
        else:    
            raise ValueError("No exact solution was provided.")
            
    def _implicit_theta_solver(self, Internal_Call = False, N = None, M = None, theta = None, **kwargs):
        """Method that solves the heat equation u_t - self.alpha * u_xx = f (t,x) with a theta-method.
        ----------------
        x0:float --- Lower bound of the spatial interval.
        xf:float --- Upper bound of the spatial interval.
        t0:float --- Lower bound of the temporall interval.
        tf:float --- Upper bound of the temporal interval
        N:int --- Number of spatial partitions.
        M:int --- Number of temporal partitions.
        alpha:float --- A real positive number.
        ua:float --- Left boundary condition.
        ub:float --- Right boundary condition.
        fun:callable --- Function defining the differential equation.
        theta:float --- Parameter which determines the method. 
                        (If theta is 1 it is the implicit line method, if theta is 0.5 it is the Crank-Nicolson method.)
        left:bool --- If left is True the left boundary condition is Neumann, if left is False the left boundary condition is Dirichlet.
        right:bool --- If right is True the right boundary condition is Neumann, if right is False the right boundary condition is Dirichlet.
        exact:callable (optional) --- The exact solution of the problem.
        ----------------
        """
        # Inicialization.
        if N == None:
            raise ValueError("N (number of partitions must be provided.")
        N = int(N)
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        if theta == None:
            raise ValueError("Parameter theta must be provided.")
        theta = float(theta)
        if (theta < 0 or theta > 1):  # We check that theta gives a weighed mean.
            raise ValueError("Theta fuera del rango.")     
        if M == None:
            raise ValueError("M (number of partitions must be provided.")
        if(theta < 0.5):  # Whenever the method is conditionally estable M is modified.
            M = np.ceil(2 * (self.tf - self.t0) * (1 - 2 * theta) / dx2)
        M = int(M)
        dt = (self.tf - self.t0) / float(M)
        t = np.linspace(self.t0, self.tf, M + 1)
        t1 = time.time()
        b = self.fun(t[0],x)
        
        #Clasification.
        if self.left:  # Left is Neumann.
            # Change first row of D.
            D[0, 1] = 2 * D[0, 1]

        else: 
            D[0, 0] = 0.0
            D[0, 1] = 0.0

        if self.right:  # Right is Neumann.
            D[N, N - 1] = 2 * D[N, N - 1]

        else:  
            D[N, N] = 0.0
            D[N, N - 1] = 0.0
            
        # Change from lil to csc for performance in matrix calculation.
        D = D.tocsc()
        AE = Id - (1-theta) * self.alpha * dt/dx2 * D  # Explicit method matrix.
        AI = Id + theta * self.alpha * dt/dx2 * D  # Implicit method matrix.
        
        LU = splu(AI)
        
        #Solution of the system.
        #Initialization of node vector.
        usol = self.u0(x)
        
        for n in range(M):
            b = dt * (theta * self.fun(x,t[n+1]) + (1-theta) * self.fun(x,t[n])) + AE * usol 
            # Change first and last entries of b depending on whether we have Neumann (True) or Dirichlet (False) conditions.
            b[0] = int(not self.left) * self.ua(t[n+1]) + int(self.left) * (b[0] - 2 * self.alpha * dt/dx * (theta * self.ua(t[n+1]) + (1 - theta) * self.ua(t[n]))) 
            b[N] = int(not self.right) * self.ub(t[n+1]) + int(self.right) * (b[N] + 2 * self.alpha * dt/dx * (theta * self.ub(t[n+1]) + (1 - theta) * self.ub(t[n])))  
            usol = LU.solve(b)
        
        #We check the time and return the solution.
        tf = time.time()  
        if not Internal_Call:
            print("Tiempo de ejecucion:", format(tf - t1))
        return t, x, usol
    
    def _explicit_solver(self, Internal_Call = False, N = None, M = None, **kwargs):
        """Method that solves the heat equation u_t - self.alpha * u_xx = f (t,x) with a the explicit line method.
        ----------------
        x0:float --- Lower bound of the spatial interval.
        xf:float --- Upper bound of the spatial interval.
        t0:float --- Lower bound of the temporall interval.
        tf:float --- Upper bound of the temporal interval
        N:int --- Number of spatial partitions.
        M:int --- Number of temporal partitions.
        alpha:float --- A real positive number.
        ua:float --- Left boundary condition.
        ub:float --- Right boundary condition.
        fun:callable --- Function defining the differential equation.
        left:bool --- If left is True the left boundary condition is Neumann, if left is False the left boundary condition is Dirichlet.
        right:bool --- If right is True the right boundary condition is Neumann, if right is False the right boundary condition is Dirichlet.
        exact:callable (optional) --- The exact solution of the problem.
        ----------------
        """
        # Initialization.
        if N == None:
            raise ValueError("N (number of partitions must be provided.")
        N = int(N)
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        if M == None:
            raise ValueError("M (number of partitions must be provided.")
        M = int(M)
        dt = (self.tf - self.t0) / float(M)
        t = np.linspace(self.t0, self.tf, M + 1)
        t1 = time.time()
        
        #Clasification.
        if self.left:  # Left is Neumann.
            # Change first row of D.
            D[0, 1] = 2 * D[0, 1]
            
        else: 
            D[0, 0] = 0.0
            D[0, 1] = 0.0

        if self.right:  # Right is Neumann.
            D[N, N - 1] = 2 * D[N, N - 1]

        else:
            D[N, N] = 0.0
            D[N, N - 1] = 0.0
            
        # Change from lil to csc for performance in matrix calculation.
        D = D.tocsc()
        A = Id - self.alpha * dt/dx2 * D
        
        #Solution of the system.
        #Initialization of node vector.
        usol = self.u0(x)

        for n in range(M):
            usol = A * usol + dt * self.fun(x,t[n]) 
            # Change first and last entries of usol.
            usol[0] = int(not self.left) * (- 2 * self.alpha * dt/dx * self.ua(t[n+1])) + int(self.left) * usol[0] 
            usol[-1] =  int(not self.right) * (2 * self.alpha * dt/dx * self.ub(t[n+1])) + int(self.right) * usol[0]
     
        #We check the time and return the solution.
        tf = time.time()  
        if not Internal_Call:
            print("Tiempo de ejecucion:", format(tf - t1))
        return t, x, usol
    
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
            
        tf = time.time()  # CWe chek the time.
        if not Internal_Call:
            print("Tiempo de ejecucion:", format(tf - t1))
            print("Número de iteraciones:", cont)
            
        return x, u_new
            
    
#Now we check the method with multiple calls which allow to empirically calculate its order.
def f(x):
    """Function which defines the ODE we want to solve."""
    y = 2.0 * np.sin(x)
    return y

def exact(x):
    """The exact solution of the problem given the Dirichlet conditions u(0) = u(pi) = 0 and alpha = 1"""
    y = np.sin(x)
    return y

equation = BVPSolver(0,np.pi,1,0,-1,f,False,True,exact)  #We call an object from BVPSolver

mesh = [50, 100, 200, 400]
error0 = 1

print("Direct Boundary.")
plt.figure("No Penalization: Numerical Solution (Blue) and Exact Solution (Red)")
for N in mesh:
    error1 = equation.solution_error(N = N)
    if N > 50:
        print(f"With N = {N} the order is = {np.log(error0 / error1)/np.log(2)} \n")
    else:
        print(f"With N = {50} the order is = --- \n")
    error0 = error1
    
def f1(x,t):
    """Functions wich defines equation 1."""
    y = x * np.cos(x * t) + t**2 * np.sin(x * t)
    return y

def exact1(x,t):
    """Exact solution of equation 1"""
    y = np.sin(x * t)
    return y

def ub(t):
    y = np.sin(t)
    return y

def ua(t):
    y = t
    return y

def u0(x):
    return 0

u0 = np.vectorize(u0)

equation2 = HTLSolver(0, 1, 0, 1, 1, u0, ua, ub, f1, True, False, exact1) #We call an object from HLTSolver

print("\n")
print("Penalization Direct Boundary.")
mesh = [5,10,20,40]
plt.figure("Penalization: Numerical Solution (Blue) and Exact Solution (Red)")
for N in mesh:
    error1 = equation2.solution_error(N = N, M = 1e4, theta = 0.5)
    if N > 5:
        print(f"With N = {N} the order is = {np.log(error0 / error1)/np.log(2)} \n")
    else:
        print(f"With N = {5} the order is = --- \n")
    error0 = error1
    
#We can also plot the solution.

equation.solution_plot(N = 400)
plt.show()