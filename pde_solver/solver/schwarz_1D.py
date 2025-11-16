import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu

from solver.base import Solver


class schwarz_k(Solver):
    """Schwarz domain decomposition solver for 1D problems.
    
    Solves equations of the form u - alpha * u'' = f using finite difference
    scheme with Schwarz domain decomposition technique. The interval is
    decomposed into k subintervals with overlaps.
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
        """Initialize Schwarz solver.
        
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

    def solver(
        self,
        Internal_Call: bool = False,
        N: Optional[int] = None,
        l: Optional[List[int]] = None,
        tol: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using Schwarz domain decomposition.
        
        Interval is decomposed into k subintervals with overlaps specified by l.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            N: Total number of partitions.
            l: Vector with entry l[i] corresponding to the number of common nodes
               of subintervals I_i and I_{i+1}.
            tol: Error tolerance for the iterative method.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (x, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided.
        """
        if N is None:
            raise ValueError("N (number of partitions) must be provided.")
        if l is None:
            raise ValueError("l (overlap vector) must be provided.")
        if tol is None:
            raise ValueError("tol (error tolerance) must be provided.")
        N = int(N)
        k = len(l)  # Number of overlaps (it actually equals k-1).
        iter_max = 500  # Maximum number of iterations.
        t1 = time.time()
        dx, dx2, x, D, Id = self._mesh_and_sys_setting(N)
        q = []  # List with subinterval nodes.
        p = []  # List with subinterval lengths.

        for i in range(1, k+1):
            q.append(i * (N + N % (k+1)) /
                     (k+1) - 0.5 * (l[i-1] + l[i-1] % 2))
            q.append(i * (N + N % (k+1)) /
                     (k+1) + 0.5 * (l[i-1] + l[i-1] % 2))

        # Index must be an int.
        q = [int(x) for x in q]

        p.append(N - q[-2])

        for i in range(1, k):
            p.append(q[2*i+1] - q[2*(i-1)])

        p.append(N - q[-2])
        p = [int(y) for y in p]

        # Define a list for each problem solve matrix
        A = []
        LU = []

        for i in range(k+1):
            D = lil_matrix((p[i] + 1, p[i] + 1), dtype="float64")
            A.append(D)
            # Access to A[i] as the last element of A.
            A[-1].setdiag(2.0 * np.ones(p[i] + 1), 0)
            A[-1].setdiag(-1.0 * np.ones(p[i]), 1)
            A[-1].setdiag(-1.0 * np.ones(p[i]), -1)
            A[-1][p[i], p[i]] = 0.0
            A[-1][p[i], p[i] - 1] = 0.0

            # Classification.
            if i == 0:  # First subdomine.
                if self.left:  # Left is Neumann.
                    A[-1][0, 1] = 2 * A[-1][0, 1]
                else:  # Left is Dirichlet.
                    A[-1][0, 0] = 0.0
                    A[-1][0, 1] = 0.0
                A[-1][p[i], p[i]] = 0.0
                A[-1][p[i], p[i] - 1] = 0.0

            elif i == k:  # Last subdomine.
                if self.right:  # Right is Neumann.
                    A[-1][p[i], p[i] - 1] = 2 * A[-1][p[i], p[i] - 1]
                else:  # Right is Dirichlet.
                    A[-1][p[i], p[i]] = 0.0
                    A[-1][p[i], p[i] - 1] = 0.0
                A[-1][0, 0] = 0.0
                A[-1][0, 1] = 0.0
            else:  # Middle subdomains.
                A[-1][0, 0] = 0.0
                A[-1][0, 1] = 0.0
                A[-1][p[i], p[i]] = 0.0
                A[-1][p[i], p[i] - 1] = 0.0

            Id = identity(p[i] + 1, dtype="float64", format="csc")
            # Use csc for performance.
            A[-1].tocsc()
            A[-1] = Id + self.alpha/dx2 * A[-1]

            # LU decomposition.
            LU.append(splu(A[-1]))

        # Resolution.
        # Horizontally stack the solution vectors of each problem.
        u_old = np.zeros(N+1)
        u_new = np.zeros(N+1)

        fun_x = self.fun(x)

        error = tol + 1
        cont = 0  # Iteration counter.
        while (error >= tol and cont < iter_max):
            # Empty the error list on each iteration.
            error_list = []
            b = fun_x.copy()  # Copy fun to b.

            # Solve first interval.
            b[0] = int(not self.left) * self.ua + int(self.left) * \
                (fun_x[0] - self.ua * 2 * self.alpha / dx)
            b[q[1]] = u_old[q[1]]
            u_new[:q[1]+1] = LU[0].solve(b[:q[1]+1])
            error_list.append(max(abs(u_new[:q[1]+1] - u_old[:q[1]+1])))
            u_old[:q[1]+1] = u_new[:q[1]+1]

            # Solve middle intervals.
            for i in range(1, k):
                b[q[2*(i-1)]] = u_old[q[2*(i-1)]]
                b[q[2*i+1]] = u_old[q[2*i+1]]
                u_new[q[2*(i-1)]:q[2*i+1] +
                      1] = LU[i].solve(b[q[2*(i-1)]:q[2*i+1]+1])
                error_list.append(
                    max(abs(u_new[q[2*(i-1)]:q[2*i+1]+1] - u_old[q[2*(i-1)]:q[2*i+1]+1])))
                u_old[q[2*(i-1)]:q[2*i+1]+1] = u_new[q[2*(i-1)]:q[2*i+1]+1]

            # Solve last interval.
            b[q[-2]] = u_old[q[-2]]
            b[-2] = int(not self.right) * self.ub + int(self.right) * \
                (fun_x[-1] - self.ub * 2 * self.alpha / dx)
            u_new[q[-2]:] = LU[-1].solve(b[q[-2]:])
            error_list.append(max(abs(u_new[q[-2]:] - u_old[q[-2]:])))
            u_old[q[-2]:] = u_new[q[-2]:]

            error = max(error_list)

            cont += 1

        if (cont == iter_max):
            print("Maximum number of iterations (", iter_max, ") was reached.")

        tf = time.time()  # We check the time.
        if not Internal_Call:
            print("Running time:", format(tf - t1))
            print("Iterations count:", cont)

        return x, u_new
