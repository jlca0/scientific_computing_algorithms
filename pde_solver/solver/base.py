import time
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import identity, lil_matrix, csc_matrix
from scipy.sparse.linalg import splu


class Solver:
    """Base class for PDE/ODE solvers.
    
    Provides common functionality for solving differential equations with
    finite difference methods using sparse matrices.
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
        """Initialize the solver.
        
        Args:
            x0: Lower bound of the spatial interval.
            xf: Upper bound of the spatial interval.
            alpha: Diffusion coefficient (must be positive).
            ua: Left boundary condition value.
            ub: Right boundary condition value.
            fun: Function defining the ODE/PDE (callable).
            left: If True, left boundary condition is Neumann; if False, Dirichlet.
            right: If True, right boundary condition is Neumann; if False, Dirichlet.
            exact: Optional exact solution for error computation and plotting.
        """
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
        
    def solver(
        self,
        Internal_Call: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the differential equation.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            **kwargs: Additional solver-specific parameters.
            
        Returns:
            Tuple of (x, solution) arrays. Base implementation returns (None, None).
        """
        return None, None
        
    def solution_plot(
        self,
        **kwargs
    ) -> None:
        """Plot numerical and exact solutions.
        
        Args:
            **kwargs: Parameters passed to solver method.
            
        Raises:
            ValueError: If no exact solution was provided.
        """
        if self.exact is not None:
            x, usol = self.solver(Internal_Call=True, **kwargs)
            plt.plot(x, usol, "b", x, self.exact(x), "r")
        else:
            raise ValueError("No exact solution was provided.")
    
    def solution_error(
        self,
        **kwargs
    ) -> float:
        """Compute and print discretization error.
        
        Args:
            **kwargs: Parameters passed to solver method.
            
        Returns:
            Maximum absolute error between numerical and exact solutions.
            
        Raises:
            ValueError: If no exact solution was provided.
        """
        if self.exact is not None:
            x, usol = self.solver(Internal_Call=True, **kwargs)
            err = max(abs(usol - self.exact(x)))
            print(f"Discretization error: {format(err)}")
            return err
        else:
            raise ValueError("No exact solution was provided.")
        
    def _validate_input(self) -> None:
        """Validate input parameters.
        
        Raises:
            ValueError: If interval is degenerate or alpha is non-positive.
        """
        if self.x0 >= self.xf:
            raise ValueError("Degenerate interval.")
        elif self.alpha <= 0:
            raise ValueError("Alpha must be positive.")
            
    def _mesh_and_sys_setting(
        self, N: int
    ) -> Tuple[float, float, np.ndarray, lil_matrix, csc_matrix]:
        """Calculate mesh and construct system matrix.
        
        Args:
            N: Number of spatial partitions.
            
        Returns:
            Tuple containing:
                - dx: Spatial step size
                - dx2: Square of spatial step size
                - x: Spatial grid points
                - D: Second derivative matrix (LIL format)
                - Id: Identity matrix (CSC format)
        """
        dx = (self.xf - self.x0) / N
        dx2 = dx * dx
        x = np.linspace(self.x0, self.xf, N + 1)
        D = lil_matrix((N + 1, N + 1), dtype="float64")
        D.setdiag(2.0 * np.ones(N + 1), 0)
        D.setdiag(-1.0 * np.ones(N), 1)
        D.setdiag(-1.0 * np.ones(N), -1)
        
        Id = identity(N + 1, dtype="float64", format="csc")
        
        return dx, dx2, x, D, Id
    