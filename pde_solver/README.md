Solvers for several ordinay and partial differential equations (ODEs and PDEs) are implemented making use of **sparse matrices techniques**, among them:

* **bvp.py** A boundary value problem of the form u(x) - alpha \* u''(x) = f(x) with Dirichlet, Neumann or mixed boundary conditions in some interval (a,b). A second-order approximation for the second derivative is applied to deduce a second order finite difference method. An use case can be found in **/examples/test\_bvp.py**
* **heat.py** The heat equation u\_t(x,t) - nu \* u\_xx(x,t) = f(x,t) withDirichlet, Neumann or mixed boundary conditions in some interval (x0,xf) \* (t0,tf). A theta-methos is implemented with theta = 0 for the explicit euler method discretization in time and theta = 1 for the explicit euler method discretization in time. However, a dedicated method is applied for the case theta = 0 for performance considerations.An use case can be found in **/examples/test\_heat.py**
* **nonlinear.py** Quadratic versions of both equations given by -u^2(x) - alpha \* u''(x) = f(x) and u\_t(x,t) - nu \* u\_xx(x,t) = u(x,t)^2 + f(x,t) with Dirichlet bundary conditions. In the first case a fixed point method and Newton method are implemented. In the second case an implicit euler method is used for time discretization together with a fixed point method. An use case can be found in **/examples/test\_nonlinear.py**

Every solver is implemented as a class inheriting from the Solver class defined in **base.py** equipped with a solver() method. For user guidelines one should refer to the **/examples** folder where *Jupyter* notebooks are provided. For a detailed mathematical justification of the methods applied one should refer to the **/documentation** folder where both a *.tex* and *.pdf* file are provided.

Requirements for running the modules are specified in **requirements.txt**.

