import numpy as np

from ..optimization.optimization_quantity import Unknown
from ..optimization.least_squares import least_squares

from .calibration_problem import CalibrationProblem


def solve(problem: CalibrationProblem):
    "Solve the given calibration problem instance via the algorithm described in README/algorithm-overview"

    ########################################################################
    # Step 1: Roughly estimate camera poses using initial-guess model (PnP)
    ########################################################################

    # TODO

    ########################################################################
    # Step 2: Jointly optimize poses and camera model params
    ########################################################################

    # collect unknowns
    unknowns = problem.collect_unknowns()

    # vector of free variables, filled in with initial guesses
    x0 = np.array([u.value() for u in unknowns])

    # residuals function
    def fun(x):
        # update underlying problem instance with candidate solution
        for u, xi in zip(unknowns, x):
            u.set_value(xi)

        return problem.get_residuals()

    # run the optimizer
    res = least_squares(fun, x0)

    print("Optimization Result:", res)

    # write back final solution
    for u, xi in zip(unknowns, res["x"]):
        u.set_value(xi)

    return res
