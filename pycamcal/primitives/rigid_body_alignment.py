import numpy as np

def estimate_rigid_body_alignment(source: np.ndarray, target: np.ndarray):
    """
    Estimate the rigid body transformation `(R, t)` that best aligns `source` to `target`:

    `R @ source + t == target`

    via the Kabsch-Umeyama algorithm.

    Note: expects column vectors (i.e. arrays should have shape `(..., 3, N)`)

    Supports batch evaluation, provided inputs are broadcastable, e.g:
     - `(3, N)` and `(3, N)`
     - `(B, 3, N)` and `(3, N)`
     - `(3, N)` and `(B, 3, N)`
     - `(B, 3, N)` and `(B, 3, N)`
    """

    assert(source.shape[-2] == target.shape[-2] == 3)
    assert(source.shape[-1] == target.shape[-1])
    assert(np.all(np.isfinite(source)))
    assert(np.all(np.isfinite(target)))

    source_centroid = np.mean(source, axis=-1, keepdims=True)
    target_centroid = np.mean(target, axis=-1, keepdims=True)

    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered @ target_centered.swapaxes(-2, -1)

    U, _, VT = np.linalg.svd(covariance)

    UT = U.swapaxes(-2, -1)
    V = VT.swapaxes(-2, -1)

    # handle case where SVD produces a reflection matrix
    M = np.zeros_like(U)
    M[...,0,0] = 1
    M[...,1,1] = 1
    M[...,2,2] = np.linalg.det(V @ UT)

    R = V @ M @ UT
    t = target_centroid - R @ source_centroid

    return R, t
