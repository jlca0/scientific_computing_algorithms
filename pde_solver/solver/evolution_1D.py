from base import Solver
import time
import numpy as np
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt


class HTLSolver(Solver):
    """Given the heat equation u_t - alpha * u_xx = fun it is numerically
    solved by the implicit lines method. The running time is measured and
    printed. The error of the stimation is also printed."""

    def __init__(self, x0, xf, t0, tf, alpha, u0: callable, ua: callable, ub: callable, fun, left, right, exact=None):
        """ --- x0, xf (float) --- Interval where the ODE is solved.
            --- ua, ub (float) --- Boundary conditions.
            --- fun (callable) --- Function which defines the ODE.
            --- left (bool) --- If TRUE, the left boundary condition is
            Neumann. If FALSE it is Dirichlet.
            --- right (bool) --- The same applies for the right boundary condition.
            --- exact (callable, optional) --- Exact solution of the problem.
            If given it automatically plots both the exact solution and the
            numerical solution"""
        super().__init__(x0, xf, alpha, 0, 0, fun, left, right, exact)
        self.t0 = t0
        self.tf = tf
        self.u0 = u0
        self.ua = ua
        self.ub = ub

    def solver(self, Internal_Call=False, N=None, M=None, theta=None, **kwargs):
        if (theta == 1):
            return self._explicit_solver(Internal_Call=True, N=N, M=M, theta=theta)
        else:
            return self._implicit_theta_solver(Internal_Call=True, N=N, M=M, theta=theta)

    def solution_plot(self, N=None, M=None, theta=None, **kwargs):
        if (self.exact is not None):
            t, x, usol = self.solver(
                Internal_Call=True, N=N, M=M, theta=theta, **kwargs)
            # We graph the last time iteration.
            plt.plot(x, usol, "b", x, self.exact(x, t[-1]), "r")
        else:
            raise ValueError("No exact solution was provided.")

    def solution_error(self, N=None, M=None, theta=None, **kwargs):
        if (self.exact is not None):
            t, x, usol = self.solver(
                Internal_Call=True, N=N, M=M, theta=theta, **kwargs)
            err = max(abs(usol - self.exact(x, t[-1])))
            print(f"Discretization error: {format(err)}")
            return err
        else:
            raise ValueError("No exact solution was provided.")

    def _implicit_theta_solver(self, Internal_Call=False, N=None, M=None, theta=None, **kwargs):
        """Method that solves the heat equation u_t - alpha * u_xx = f(t,x)
        with a theta-method.
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
        # We check that theta gives a weighed mean.
        if (theta < 0 or theta > 1):
            raise ValueError("Theta fuera del rango.")
        if M == None:
            raise ValueError("M (number of partitions must be provided.")
        # Whenever the method is conditionally estable M is modified.
        if (theta < 0.5):
            M = np.ceil(2 * (self.tf - self.t0) * (1 - 2 * theta) / dx2)
        M = int(M)
        dt = (self.tf - self.t0) / float(M)
        t = np.linspace(self.t0, self.tf, M + 1)
        t1 = time.time()
        b = self.fun(t[0], x)

        # Clasification.
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
        # Explicit method matrix.
        AE = Id - (1-theta) * self.alpha * dt/dx2 * D
        AI = Id + theta * self.alpha * dt/dx2 * D  # Implicit method matrix.

        LU = splu(AI)

        # Solution of the system.
        # Initialization of node vector.
        usol = self.u0(x)

        for n in range(M):
            b = dt * (theta * self.fun(x, t[n+1]) +
                      (1-theta) * self.fun(x, t[n])) + AE * usol
            # Change first and last entries of b depending on whether we have Neumann (True) or Dirichlet (False) conditions.
            b[0] = int(not self.left) * self.ua(t[n+1]) + int(self.left) * (b[0] - 2 *
                                                                            self.alpha * dt/dx * (theta * self.ua(t[n+1]) + (1 - theta) * self.ua(t[n])))
            b[N] = int(not self.right) * self.ub(t[n+1]) + int(self.right) * (b[N] + 2 *
                                                                              self.alpha * dt/dx * (theta * self.ub(t[n+1]) + (1 - theta) * self.ub(t[n])))
            usol = LU.solve(b)

        # We check the time and return the solution.
        tf = time.time()
        if not Internal_Call:
            print("Tiempo de ejecucion:", format(tf - t1))
        return t, x, usol

    def _explicit_solver(self, Internal_Call=False, N=None, M=None, **kwargs):
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

        # Clasification.
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

        # Solution of the system.
        # Initialization of node vector.
        usol = self.u0(x)

        for n in range(M):
            usol = A * usol + dt * self.fun(x, t[n])
            # Change first and last entries of usol.
            usol[0] = int(not self.left) * (- 2 * self.alpha *
                                            dt/dx * self.ua(t[n+1])) + int(self.left) * usol[0]
            usol[-1] = int(not self.right) * (2 * self.alpha *
                                              dt/dx * self.ub(t[n+1])) + int(self.right) * usol[0]

        # We check the time and return the solution.
        tf = time.time()
        if not Internal_Call:
            print("Tiempo de ejecucion:", format(tf - t1))
        return t, x, usol
