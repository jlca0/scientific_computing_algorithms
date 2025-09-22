import time
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
import matplotlib.pyplot as plt


def direct_boundary(x0, xf, N, alpha, ua, ub, fun, left, right, exact = None):
    """Given a boundary conditions problem of the form u - alpha * u'' = fun it is numerically solved by a finite
    difference scheme. The running time is measured and printed. The error of the stimation is also printed.
    
    x0, xf --- Interval where the ODE is solved.
    N --- Number of partitions of the mesh. 
    ua, ub --- Boundary conditions.
    fun --- Function which defines the ODE.
    left --- If left = TRUE then the left boundary condition is Neumann, if left = FALSE it is Dirichlet. 
    right --- The same applies for the right boundary condition.
    exact --- Exact solution of the problem. If given it automatically plots both the exact solution and the numerical solution"""
    
    # Inicialization. Data input verification.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    alpha = float(alpha)
    ua = float(ua)
    ub = float(ub)
    left = bool(left)
    right = bool(right)
    x = np.linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construction of D, without taking into account the first and last equations.
    D.setdiag(2.0 * np.ones(N + 1), 0)
    D.setdiag(-1.0 * np.ones(N), 1)
    D.setdiag(-1.0 * np.ones(N), -1)
    # Construction of the independent term b.
    b = fun(x)

    # Clasification of the boundary conditions.
    if left and right:  # Both Neumann.
        # The first and last equations are changed.
        D[N, N - 1] = 2 * D[N, N - 1]
        D[0, 1] = 2 * D[0, 1]
        
        # The first and last entry of b are changed.
        b[0] -= ua * 2 * alpha / dx
        b[N] += ub * 2 * alpha / dx

    elif left:  # Left Neumann.
        D[N, N] = 0.0
        D[N, N - 1] = 0.0
        D[0, 1] = 2 * D[0, 1]

        b[0] -= ua * 2 * alpha / dx
        b[N] += ub

    elif right:  # Right Neumann.
        D[0, 0] = 0.0
        D[0, 1] = 0.0
        D[N, N - 1] = 2 * D[N, N - 1]

        b[0] += ua
        b[N] += ub * 2 * alpha / dx

    else:  # Both Dirichlet.
        D[0, 0] = 0.0
        D[0, 1] = 0.0
        D[N, N] = 0.0
        D[N, N - 1] = 0.0

        b[0] = ua
        b[N] = ub
    # Change from lil to csc for performance in matrix calculation.
    D = D.tocsc()
    # Notice that the matrix A is not symmetric.(Cholesky does not apply.)
    A = Id + alpha / dx2 * D
    
    # Solution of the system.
    # Complete LU factorization of the sparse matrix A. 
    LU = splu(A)
    usol = LU.solve(b)

    #We check the time and print the results.
    tf = time.time()
    print(f"Runnign time: {format(tf - t1)}")
    plt.plot(x, usol, "b")
    #Both the plotting and the error display are ignored if no exact solution was given as input.
    if exact != None: 
        plt.plot(x, exact(x), "r")
        err = max(abs(usol - exact(x)))
        print(f"Discretization error: {format(err)}")
        return err

#Now we check the method with multiple calls which allow to empirically calculate its order.
def f(x):
    """Function which defines the ODE we want to solve."""
    y = 2.0 * np.sin(x)
    return y

def uexact(x):
    """The exact solution of the problem given the Dirichlet conditions u(0) = u(pi) = 0 and alpha = 1"""
    y = np.sin(x)
    return y

mesh = [50, 100, 200, 400]
error0 = 1

plt.figure("Numerical Solution (Blue) and Exact Solution (Red)")
for N in mesh:
    error1 = direct_boundary(0, np.pi, N, 1, 0, -1, f, False, True, uexact)
    if N > 50:
        print(f"With N = {N} the order is = {0.5 * error0 / error1} \n")
    else:
        print(f"With N = {50} the order is = --- \n")
    error0 = error1