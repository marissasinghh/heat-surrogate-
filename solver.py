##  Solver.py - The Finite Difference Solver for the 1D Heat Equation
## ------------------------------------------------------------------
##  This module uses a hand derived forward-time centered-space scheme 
#   (FTCS Scheme) of the heat equation PDE to approximate solutions 
##  that take thousands of tiny steps forward in time 
##  i.e. U(x, t+dt) = FTCS Scheme 
##
##  Solves u_t = α·u_xx on x ∈ [0, 1] with u(0,t) = u(1,t) = 0, 
#   using FTCS scheme:

##  u(x, t+dt) = u(x, t) + r·[u(x+dx, t) − 2u(x, t) + u(x−dx, t)],  r = α·dt/dx²

##  Stable for r ≤ 0.5. Verified against the exact sine-mode solution.

##  Role in the project:
##    - answer key: every model (PINN, FNO) is scored against it
##    - data factory: generates the training runs for the FNO
##    - speed benchmark: the time a surrogate has to beat


import numpy as np
from constants import A_K_LOWER_BOUND, A_K_UPPER_BOUND, X_FINAL, N_MODES, NUMBER_SPATIAL_GRID_POINTS

# Using vectorized array slicing to calculate spatial derivatives 

# Functions to be moved to helper function file
# ---------------------------------------------


def build_x_spatial_grid(nx:int, L:float) -> tuple[np.ndarray, float]:
    """Build a uniform spatial grid on [0, L].

    Parameters
    ----------
    nx : int
        Number of grid points, including both boundaries.
    L : float
        Domain length.

    Returns
    -------
    x : np.ndarray, shape (nx,)
        Grid point positions.
    dx : float
        Spacing between points, L / (nx - 1).
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx) 
    
    return x, dx


def generate_random_coefficients(random_number_generator, lower_bound:float, upper_bound:float, n_modes:int) -> np.ndarray:
    """Draw random sine-mode amplitudes a_k for one initial condition.

    Parameters
    ----------
    random_number_generator : np.random.Generator
        Seeded generator, passed in so runs are reproducible.
    lower_bound, upper_bound : float
        Each a_k is drawn uniformly from [lower_bound, upper_bound).
    n_modes : int
        Number of modes (k = 1 ... n_modes).

    Returns
    -------
    a_k : np.ndarray, shape (n_modes,)
        a_k[i] is the amplitude of mode k = i + 1.
    """
    coefficients = random_number_generator.uniform(lower_bound, upper_bound, n_modes)
    
    return coefficients


def build_initial_condition(a_k: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Build u(x, 0) as a sum of sine modes.

    u0(x) = Σ_k a_k · sin(kπx),  k = 1 ... len(a_k)

    Each mode is zero at x = 0 and x = 1, so u0 satisfies the
    boundary conditions u(0) = u(1) = 0.

    Parameters
    ----------
    a_k : np.ndarray, shape (n_modes,)
        Mode amplitudes; a_k[i] multiplies sin((i+1)πx).
    x : np.ndarray, shape (nx,)
        Grid points on [0, 1].

    Returns
    -------
    u0 : np.ndarray, shape (nx,)
        Initial temperature profile, with endpoints exactly 0.
    """
    u0 = np.zeros_like(x)
    for i in range(0, len(a_k)):
        k = i + 1
        u0 += a_k[i]*np.sin(k*np.pi*x)

    # Forcing boundary conditions
    u0[0] = 0.0 
    u0[-1] = 0.0

    return u0


def ftcs_step(u: np.ndarray, r: float) -> np.ndarray:
    """Advance the temperature profile one time step with FTCS.

    Parameters
    ----------
    u : np.ndarray, shape (nx,)
        Temperature at every grid point at the current time,
        boundaries included (u[0] = u[-1] = 0).
    r : float
        Dimensionless step size alpha·dt/dx². Must be ≤ 0.5 for stability.

    Returns
    -------
    u_new : np.ndarray, shape (nx,)
        Temperature one dt later. Boundaries are left unchanged.
        The input array is not modified.

    Notes
    -----
    Interior update, from forward difference in t and central
    difference in x:

        u(x, t+dt) = u(x, t) + r·[u(x+dx, t) - 2u(x, t) + u(x-dx, t)]

    Rewritten as (1 - 2r)·u(x) + r·u(x+dx) + r·u(x-dx), this is a
    weighted average. r > 0.5 makes the (1 - 2r) weight negative,
    which causes the growing sawtooth instability.
    """
    u_new = u.copy() 
    u_new[1:-1] = u[1:-1]+r*(u[2:]-2*u[1:-1]+u[:-2])
    
    return u_new


# Main.py begin
# -------------

if __name__ == "__main__":

    rng = np.random.default_rng(seed=42) 

    x, dx = build_x_spatial_grid(NUMBER_SPATIAL_GRID_POINTS, X_FINAL)
    a_k = generate_random_coefficients(rng, A_K_LOWER_BOUND, A_K_UPPER_BOUND, N_MODES)
    u0 = build_initial_condition(a_k, x)



    


# EXAMPLE 
# f = x**2                                    # some sampled function
# dfdx = (f[2:] - f[:-2]) / (2 * dx)          # central 1st derivative, interior only
# f[2:]  → "the point to the right" for each interior point
# f[:-2] → "the point to the left"
# result has length nx - 2 (no value at the two edges)

# dudx = (u[2:] - u[:-2]) / (2 * dx)

 



