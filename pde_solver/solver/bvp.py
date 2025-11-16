import time
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import splu

from solver.base import Solver


class BVPSolver(Solver):
    """Solver for boundary value problems of the form u - alpha * u'' = fun.
    
    Uses finite difference scheme with second-order accuracy. Supports
    Dirichlet, Neumann, or mixed boundary conditions. The running time
    is measured and printed. The error of the estimation is also printed.
    """

    def __init__(
        self,
        x0: float,
        xf: float,
        alpha: float,
        ua: float,
        ub: float,
        fun: Callable,
        left: bool,
        right: bool,
        exact: Optional[Callable] = None
    ) -> None:
        """Initialize BVP solver.
        
        Args:
            x0: Lower bound of the spatial interval.
            xf: Upper bound of the spatial interval.
            alpha: Diffusion coefficient (must be positive).
            ua: Left boundary condition value.
            ub: Right boundary condition value.
            fun: Function defining the ODE (callable).
            left: If True, left boundary condition is Neumann; if False, Dirichlet.
            right: If True, right boundary condition is Neumann; if False, Dirichlet.
            exact: Optional exact solution for error computation and plotting.
        """
        super().__init__(x0, xf, alpha, ua, ub, fun, left, right, exact)

    def solver(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the boundary value problem.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Number of partitions of the mesh.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (x, solution) arrays.
            
        Raises:
            ValueError: If N is not provided.
        """
        if N is None:
            raise ValueError("N (number of partitions) must be provided.")
        N = int(N)
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        b = self.fun(x)

        # Classification
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

        # Change from lil to csc for performance in matrix calculation
        D = D.tocsc()
        # Notice that the matrix A is not symmetric (Cholesky does not apply)
        A = Id + self.alpha / dx2 * D

        # Solution of the system
        # Complete LU factorization of the sparse matrix A
        LU = splu(A)
        usol = LU.solve(b)

        # We check the time and return the solution
        if not Internal_Call:
            tf = time.time()
            print(f"Running time: {format(tf - t1)}")
        return x, usol
