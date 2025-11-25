import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import Any, Callable, Dict, Tuple

# WARNING: 99% AI generated

def least_squares(
    residual_fn: Callable[[Any], jnp.ndarray],
    x0: Any,
    *,
    args=(),
    maxiter: int = 100,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    lambda0: float = 1e-3,
    lambda_inc: float = 10.0,
    lambda_dec: float = 0.1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Levenberg-Marquardt solver for min_x 0.5 * ||r(x)||^2
    - residual_fn(x, *args) -> residual vector (1D)
    - x0: initial params as pytree (arrays, tuple, dict ...)
    Returns dict: {'x': optimized pytree, 'cost': final cost, 'niters', 'success', 'info'}
    """
    # flatten pytree params -> 1D vector and back
    x0_flat, unravel = ravel_pytree(x0)

    # wrapped residual that works on flat vector
    def residuals_flat(x_flat):
        x = unravel(x_flat)
        r = residual_fn(x, *args) if args else residual_fn(x)
        r = jnp.asarray(r)
        return r.reshape((-1,))  # ensure 1D

    jac_fn = jax.jacobian(residuals_flat)  # (m, n)
    res_fn = residuals_flat

    x = x0_flat
    lam = lambda0
    success = False

    for k in range(maxiter):
        r = res_fn(x)
        cost = 0.5 * jnp.dot(r, r)

        J = jac_fn(x)          # shape (m, n)
        # build normal equations
        JTJ = J.T @ J          # (n, n)
        g = J.T @ r            # (n,)

        # termination on gradient or tiny step
        g_norm = jnp.max(jnp.abs(g))    # L-inf norm
        if g_norm <= xtol:
            success = True
            if verbose:
                print(f"converged (grad) iter={k}")
            break

        # LM step: (JTJ + lam * diag(JTJ)) dp = -g
        A = JTJ + lam * jnp.diag(jnp.diag(JTJ) + 1e-16)

        # handle possible singularities (small regularizer)
        try:
            dp = jnp.linalg.solve(A, -g)
        except Exception:
            # fall back to pseudo-inverse solution
            dp = -jnp.linalg.lstsq(A, g, rcond=None)[0]

        if not jnp.all(jnp.isfinite(dp)):
            if verbose:
                print("non-finite step, stopping")
            break

        x_new = x + dp
        r_new = res_fn(x_new)
        cost_new = 0.5 * jnp.dot(r_new, r_new)

        if cost_new < cost:
            # accept step
            x = x_new
            lam = jnp.maximum(lam * lambda_dec, 1e-16)
            if jnp.abs(cost - cost_new) <= ftol * (1.0 + cost):
                success = True
                if verbose:
                    print(f"converged (cost) iter={k}")
                break
        else:
            # reject and increase damping
            lam = lam * lambda_inc

    x_opt = unravel(x)
    final_r = residuals_flat(x)
    final_cost = 0.5 * jnp.dot(final_r, final_r)
    info = {"niters": k + 1, "lam": float(lam), "g_norm_inf": float(g_norm)}

    return {"x": x_opt, "cost": float(final_cost), "success": bool(success), "info": info}
