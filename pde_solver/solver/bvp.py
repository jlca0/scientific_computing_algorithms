from solver.base import Solver
import time
from scipy.sparse.linalg import splu


class BVPSolver(Solver):
    """Given a boundary conditions problem of the form u - alpha * u'' = fun
    it is numerically solved by a finite difference scheme. The running time
    is measured and printed. The error of the stimation is also printed."""

    def __init__(self, x0, xf, alpha, ua, ub, fun, left, right, exact=None):
        """ --- x0, xf (float) --- Interval where the ODE is solved.
            --- ua, ub (float) --- Boundary conditions.
            --- fun (callable) --- Function which defines the ODE.
            --- left (bool) --- If TRUE, the left boundary condition
            is Neumann. If FALSE it is Dirichlet.
            --- right (bool) --- The same applies for the right
            boundary condition.
            --- exact (callable, optional) --- Exact solution of the problem.
            If given it automatically plots both the exact solution and the
            numerical solution"""
        super().__init__(x0, xf, alpha, ua, ub, fun, left, right, exact)

    def solver(self, Internal_Call=False, N=None, **kwargs):
        """
          N (int) --- Number of partitions of the mesh."""
        if N is None:
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

        # We check the time and return the solution.
        if not Internal_Call:
            tf = time.time()
            print(f"Runnign time: {format(tf - t1)}")
        return x, usol
