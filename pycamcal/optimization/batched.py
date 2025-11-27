import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from matplotlib import pyplot as plt
from tqdm import tqdm

def gauss_newton_batched(
    r_func,      # r(x, *args) → residuals, shape (B, M)
    J_func,      # J(x, *args) → Jacobian, shape (B, M, D)
    x0,          # initial guess, shape (B, D)
    args=(), 
    max_iters=10,
    ftol=1e-8,
    damping=1e-3, # Tikhonov regularization for stability
    verbose=False
):
    """
    Fully batched Gauss–Newton solver.
    Each batch element is solved independently.
    """

    def step(x):
        r = r_func(x, *args)         # (B, M)
        J = J_func(x, *args)         # (B, M, D)

        # Compute normal equations: J^T J Δ = J^T r
        JT = jnp.swapaxes(J, -1, -2)      # (B, D, M)
        JTJ = JT @ J                      # (B, D, D)
        JTr = JT @ r[..., None]           # (B, D, 1)

        # Damped (Levenberg-ish) stabilizer
        JTJ_reg = JTJ + damping * jnp.eye(JTJ.shape[-1])

        # Solve normal equations for each batch independently
        delta = jscipy.linalg.solve(JTJ_reg, JTr)[..., 0]  # (B, D)

        return x - delta, r

    x = x0
    for _ in tqdm(range(max_iters), disable=(not verbose), desc="Gauss-Newton iterations"):
        x_new, residual = step(x)
        x = x_new

        if jnp.all(residual < ftol):
            break

    return x
