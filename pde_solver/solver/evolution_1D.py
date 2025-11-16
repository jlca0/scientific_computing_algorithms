import time
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import splu

from solver.base import Solver


class HTLSolver(Solver):
    """Solver for heat equation u_t - alpha * u_xx = fun.
    
    Implements theta-method for time discretization with finite difference
    scheme in space. Supports explicit (theta=0) and implicit methods.
    """

    def __init__(
        self,
        x0: float,
        xf: float,
        t0: float,
        tf: float,
        alpha: float,
        u0: Callable,
        ua: Callable,
        ub: Callable,
        fun: Callable,
        left: bool,
        right: bool,
        exact: Optional[Callable] = None
    ) -> None:
        """Initialize heat equation solver.
        
        Args:
            x0: Lower bound of the spatial interval.
            xf: Upper bound of the spatial interval.
            t0: Lower bound of the temporal interval.
            tf: Upper bound of the temporal interval.
            alpha: Diffusion coefficient (must be positive).
            u0: Initial condition function (callable).
            ua: Left boundary condition function (callable, time-dependent).
            ub: Right boundary condition function (callable, time-dependent).
            fun: Function defining the PDE source term (callable).
            left: If True, left boundary condition is Neumann; if False, Dirichlet.
            right: If True, right boundary condition is Neumann; if False, Dirichlet.
            exact: Optional exact solution for error computation and plotting.
        """
        super().__init__(x0, xf, alpha, 0, 0, fun, left, right, exact)
        self.t0 = t0
        self.tf = tf
        if self.t0 >= self.tf:
            raise ValueError("Degenerate time interval: t0 must be less than tf.")
        self.u0 = u0
        self.ua = ua
        self.ub = ub

    def solver(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        M: Optional[int] = None,
        theta: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the heat equation.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of spatial partitions.
            M: Number of temporal partitions.
            theta: Theta parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit).
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (t, x, solution) arrays.
        """
        if theta == 1:
            return self._explicit_solver(Internal_Call=True, N=N, M=M, theta=theta)
        else:
            return self._implicit_theta_solver(Internal_Call=True, N=N, M=M, theta=theta)

    def solution_plot(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        theta: Optional[float] = None,
        **kwargs
    ) -> None:
        """Plot numerical and exact solutions.
        
        Args:
            N: Number of spatial partitions.
            M: Number of temporal partitions.
            theta: Theta parameter for time discretization.
            **kwargs: Additional parameters (unused).
            
        Raises:
            ValueError: If no exact solution was provided.
        """
        if self.exact is not None:
            t, x, usol = self.solver(
                Internal_Call=True, N=N, M=M, theta=theta, **kwargs
            )
            # We graph the last time iteration
            plt.plot(x, usol, "b", x, self.exact(x, t[-1]), "r")
        else:
            raise ValueError("No exact solution was provided.")

    def solution_error(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        theta: Optional[float] = None,
        **kwargs
    ) -> float:
        """Compute and print discretization error.
        
        Args:
            N: Number of spatial partitions.
            M: Number of temporal partitions.
            theta: Theta parameter for time discretization.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Maximum absolute error between numerical and exact solutions.
            
        Raises:
            ValueError: If no exact solution was provided.
        """
        if self.exact is not None:
            t, x, usol = self.solver(
                Internal_Call=True, N=N, M=M, theta=theta, **kwargs
            )
            err = max(abs(usol - self.exact(x, t[-1])))
            print(f"Discretization error: {format(err)}")
            return err
        else:
            raise ValueError("No exact solution was provided.")

    def _implicit_theta_solver(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        M: Optional[int] = None,
        theta: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the heat equation using theta-method.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of spatial partitions.
            M: Number of temporal partitions.
            theta: Theta parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit).
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (t, x, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided or theta is out of range.
        """
        # Inicialization.
        if N is None:
            raise ValueError("N (number of partitions must be provided.")
        N = int(N)
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        if theta is None:
            raise ValueError("Parameter theta must be provided.")
        theta = float(theta)
        # We check that theta gives a weighed mean.
        if (theta < 0 or theta > 1):
            raise ValueError("Theta out of range.")
        if M is None:
            raise ValueError("M (number of partitions must be provided.")
        # Whenever the method is conditionally stable M is modified.
        if (theta < 0.5):
            M = np.ceil(2 * (self.tf - self.t0) * (1 - 2 * theta) / dx2)
        M = int(M)
        dt = (self.tf - self.t0) / float(M)
        t = np.linspace(self.t0, self.tf, M + 1)
        t1 = time.time()
        b = self.fun(t[0], x)

        # Classification.
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
            print("Running time:", format(tf - t1))
        return t, x, usol

    def _explicit_solver(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        M: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the heat equation using explicit method.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of spatial partitions.
            M: Number of temporal partitions.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (t, x, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided.
        """
        # Initialization.
        if N is None:
            raise ValueError("N (number of partitions must be provided.")
        N = int(N)
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        if M is None:
            raise ValueError("M (number of partitions must be provided.")
        M = int(M)
        dt = (self.tf - self.t0) / float(M)
        t = np.linspace(self.t0, self.tf, M + 1)
        t1 = time.time()

        # Classification.
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
            print("Running time:", format(tf - t1))
        return t, x, usol
