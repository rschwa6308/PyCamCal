import numpy as np
from scipy.spatial.transform import Rotation as R3D


def get_lookat_orientation(cam_pos, target, up) -> R3D:
    """
    Get the rotation that orients a camera at `cam_pos` to look at `target`.
    Roll about the boresight axis is chosen to best match the given `up` vector.

    Camera frame convention: +x=right, +y=down, +z=forward
    """

    down_requested = -np.array(up)

    # boresight direction
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)

    # right direction (from requested up)
    right = np.cross(down_requested, fwd)

    # actual down (orthonormal)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)

    R_cam2world = np.column_stack((right, down, fwd))
    return R3D.from_matrix(R_cam2world)


def get_random_rotation(magnitude: float, degrees=False) -> R3D:
    # random axis (uniform from suface of a sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    return R3D.from_rotvec(axis * magnitude, degrees=degrees)


def is_perfect_square(n: int):
    root = int(np.sqrt(n))
    return root**2 == n


def cartesian_product(*arrays: np.ndarray, keepdims: bool = False) -> np.ndarray:
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    if keepdims:
        return arr
    else:
        return arr.reshape(-1, la)



def is_polynomial_invertible_on_interval(coeffs, a, b, tol=1e-12):
    """
    Determine whether a real polynomial is invertible on [a, b] by
    checking that the derivative never crosses zero.

    Parameters
    ----------
    coeffs : array-like
        Polynomial coefficients in descending powers (np.poly1d format).
    a, b : float
        Interval endpoints.
    tol : float
        Numerical tolerance.

    Returns
    -------
    invertible : bool
        True if polynomial is strictly monotonic on [a, b].
    monotonicity : str
        "increasing", "decreasing", or "not_monotonic"
    """

    coeffs = np.asarray(coeffs, dtype=float)

    if a >= b:
        raise ValueError("Require a < b")

    # Derivative polynomial
    dcoeffs = np.polyder(coeffs)

    # If derivative is identically zero → constant function → not invertible
    if np.all(np.abs(dcoeffs) < tol):
        return False, "not_monotonic"

    # Find real roots of derivative
    droots = np.roots(dcoeffs)
    real_roots = droots[np.abs(droots.imag) < tol].real

    # Keep only roots strictly inside (a,b)
    critical = real_roots[(real_roots > a + tol) & (real_roots < b - tol)]

    # Sample derivative on each monotonic sub-interval
    test_intervals = np.concatenate([
        [a],
        critical,
        [b]
    ])

    # Midpoints for sign testing
    mids = 0.5 * (test_intervals[:-1] + test_intervals[1:])
    dvals = np.polyval(dcoeffs, mids)

    all_pos = np.all(dvals > 0)
    all_neg = np.all(dvals < 0)

    if all_pos:
        return True, "increasing"
    elif all_neg:
        return True, "decreasing"
    else:
        return False, "not_monotonic"

