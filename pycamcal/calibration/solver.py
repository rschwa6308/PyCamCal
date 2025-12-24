import jax.numpy as jnp

from ..optimization.optimization_quantity import RESOLVE_VALUES, Unknown
from ..optimization.least_squares import least_squares

from .calibration_problem import CalibrationProblem


def solve(problem: CalibrationProblem, **kwargs):
    """
    Solve the given calibration problem instance via the algorithm described in README/algorithm-overview.
    
    Modifies the problem instance in-place iteratively.
    Also returns a copy of the resulting calibrated camera model(s).
    """

    ########################################################################
    # Step 1: Roughly estimate camera poses using initial-guess model (PnP)
    ########################################################################

    # TODO

    ########################################################################
    # Step 2: Jointly optimize poses and camera model params
    ########################################################################

    # collect unknowns
    unknowns = problem.collect_unknowns()
    print(f"Solving calibration problem with {len(unknowns)} unknowns")

    # collection of all free variables, filled in with initial guesses
    x0 = [u.value() for u in unknowns]

    # residuals function
    def fun(x):
        print(f"fun({x})")

        # update underlying problem instance with candidate solution
        for u, xi in zip(unknowns, x):
            u.set_value(xi)

        return problem.get_residuals()

    # run the optimizer
    res = least_squares(fun, x0, **kwargs)

    print("Optimization Result:", res)

    # write back final solution
    for u, xi in zip(unknowns, res["x"]):
        u.set_value(xi.item())

    ########################################################################
    # Step 3: Return results
    ########################################################################

    cmods_final = {
        cam_id: RESOLVE_VALUES(cmod)
        for cam_id, cmod in problem.cameras.items()
    }

    return cmods_final
