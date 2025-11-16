import time
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from solver.base import Solver


class QuadraticSolver(Solver):
    """Solver for quadratic nonlinear equations -nu * u'' - u^2 = fun.
    
    Implements both fixed-point iteration and Newton's method for solving
    the nonlinear boundary value problem.
    """

    def __init__(
        self,
        x0: float,
        xf: float,
        nu: float,
        ua: float,
        ub: float,
        fun: Callable,
        left: bool,
        right: bool,
        exact: Optional[Callable] = None
    ) -> None:
        """Initialize quadratic solver.
        
        Args:
            x0: Lower bound of the spatial interval.
            xf: Upper bound of the spatial interval.
            nu: Diffusion coefficient (must be positive).
            ua: Left boundary condition value.
            ub: Right boundary condition value.
            fun: Function defining the ODE (callable).
            left: If True, left boundary condition is Neumann; if False, Dirichlet.
            right: If True, right boundary condition is Neumann; if False, Dirichlet.
            exact: Optional exact solution for error computation and plotting.
        """
        super().__init__(x0, xf, nu, ua, ub, fun, left, right, exact)

    def solver_fixed_point(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        tol: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using fixed-point iteration method.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of partitions of the mesh.
            tol: Error tolerance for convergence.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (x, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided.
        """
        if N is None:
            raise ValueError("N (number of partitions) must be provided.")
        if tol is None:
            raise ValueError("tol (error tolerance) must be provided.")
        iter_max = 500  # Maximum number of iterations
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        D = self.alpha / dx2 * D
        D[0, 0] = 1.0
        D[0, 1] = 0.0
        D[N, N] = 1.0
        D[N, N - 1] = 0.0
        D = D.tocsc()
        
        u_old = np.zeros(N + 1)
        u_new = np.zeros(N + 1)
        error = tol + 1
        fun_x = self.fun(x)
        LU = splu(D)

        cont = 0  # Iteration counter
        while error >= tol and cont < iter_max:
            b = fun_x + u_old ** 2
            b[0] = self.ua
            b[-1] = self.ub
            u_new = LU.solve(b)
            error = max(abs(u_new - u_old))
            u_old = u_new
            cont += 1

        if cont == iter_max:
            print("Maximum number of iterations (", iter_max, ") was reached.")

        tf = time.time()  # We check the time
        if not Internal_Call:
            print("Running time:", format(tf - t1))
            print("Iterations count:", cont)

        return x, u_new

    def solver_newton(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        tol: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using Newton's method.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of partitions of the mesh.
            tol: Error tolerance for convergence.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (x, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided.
        """
        if N is None:
            raise ValueError("N (number of partitions) must be provided.")
        if tol is None:
            raise ValueError("tol (error tolerance) must be provided.")
        iter_max = 500  # Maximum number of iterations
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        U_diag = lil_matrix((N + 1, N + 1), dtype="float64")
        U_diag.setdiag(np.ones(N + 1), 0)
        U_diag = U_diag.tocsc()
        D = self.alpha / dx2 * D
        D[0, 0] = 1.0
        D[0, 1] = 0.0
        D[N, N] = 1.0
        D[N, N - 1] = 0.0
        D = D.tocsc()

        # Resolution
        u_old = np.zeros(N + 1)
        u_new = np.zeros(N + 1)
        error = tol + 1
        fun_x = self.fun(x)

        cont = 0  # Iteration counter
        while error >= tol and cont < iter_max:
            b = -self.alpha / dx2 * D * u_old + u_old ** 2 + fun_x
            b[0] = -u_old[0] + self.ua
            b[-1] = -u_old[-1] + self.ub
            # Jacobian matrix
            U_diag.setdiag(2 * u_old, 0)
            J = self.alpha / dx2 * D - U_diag
            J[0, 0] = 1
            J[0, 1] = 0
            J[N, N] = 1
            J[N, N - 1] = 0
            LU = splu(J)

            y_new = LU.solve(b)
            u_new = y_new + u_old
            error = max(abs(u_new - u_old))
            u_old = u_new
            cont += 1

        if cont == iter_max:
            print("Maximum number of iterations (", iter_max, ") was reached.")

        tf = time.time()  # We check the time
        if not Internal_Call:
            print("Running time:", format(tf - t1))
            print("Iterations count:", cont)

        return x, u_new
