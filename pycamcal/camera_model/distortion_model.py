from abc import ABC
from dataclasses import dataclass
from functools import lru_cache

from matplotlib import pyplot as plt
import scipy
import jax.scipy.optimize
import numpy as np
import jax.numpy as jnp
import scipy.optimize
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
from tqdm import tqdm


from ..primitives.math_helpers import cartesian_product
from ..optimization.batched import gauss_newton_batched


class DistortionModel(ABC):
    def distort(self, rays_external: jnp.ndarray) -> jnp.ndarray:
        "Apply distortion. Input/Output rays expressed in normalized coords (i.e. intersection with z=1)"

        raise NotImplementedError()

    def undistort(self, rays_internal: jnp.ndarray) -> jnp.ndarray:
        "Invert distortion. Input/Output rays expressed in normalized coords (i.e. intersection with z=1)"

        raise NotImplementedError()


class RadialTangentialDistortion(DistortionModel):
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    def __init__(self, k1, k2, k3, p1, p2):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

        self.interp_inverse = None

    def precompute_undistort_lut(self):
        """
        Precompute a lookup-table for the inverse distortion model by passing a large number of
        sample points through the forward model, and fitting an interpolator to the resulting graph (transposed). 
        """
    
        print("precompute_undistort_lut()")

        # sample points from the FOV (with buffer)
        steps_x = np.linspace(-1.5, +1.5, num=1000)
        steps_y = np.linspace(-1.5, +1.5, num=1000)
        points_raw = cartesian_product(steps_x, steps_y, keepdims=False)

        # TODO: limit sample points to region where model is known to be invertible

        # pass through forward model
        points_distorted = self.distort(points_raw)

        # fit a C1-smooth interpolator
        self.interp_inverse = CloughTocher2DInterpolator(points_distorted, points_raw, fill_value=np.nan)

    def distort(self, rays_external: jnp.ndarray) -> jnp.ndarray:
        x = rays_external[..., 0]
        y = rays_external[..., 1]
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4

        # Radial distortion
        radial = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        # plt.plot(r2, radial)
        # plt.show()

        # Tangential distortion
        x_tang = 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
        y_tang = self.p1*(r2 + 2*y**2) + 2*self.p2*x*y

        x_distorted = x*radial + x_tang
        y_distorted = y*radial + y_tang

        return jnp.column_stack((x_distorted, y_distorted))

    def undistort(self, rays_internal: jnp.ndarray) -> jnp.ndarray:
        if self.interp_inverse is None:
            self.precompute_undistort_lut()

        result = self.interp_inverse(rays_internal)
        return result

    def is_valid_over_entire_fov(self, fov_xy, degrees=False):
        """
        Ensure that for every internal point `x` in the FOV, there exists an external point `w` such that `distort(w) = x`.
        If not, then there are pixels which are effectively un-reachable.

        Currently only considers the radial component.
        """

        fov_xy = np.array(fov_xy)

        if degrees:
            fov_xy = np.deg2rad(fov_xy)
        
        # maximum distance (to the corner of the fov) in the z=1 image plane
        r_max = np.linalg.norm(np.tan(fov_xy/2))

        # Given that distort(r) = r * (1 + k1*r^2 + k2*r^4 + k3*r^6),
        # attempt to find solution to distort(r) = r_max.

        #                 r^7         r^5         r^3         r   1
        roots = np.roots([self.k3, 0, self.k2, 0, self.k1, 0, 1, -r_max])

        has_solution = np.any(np.isreal(roots) & (roots > 0))
        return has_solution
