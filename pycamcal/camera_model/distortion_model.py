from abc import ABC
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.optimize
from tqdm import tqdm


class DistortionModel(ABC):
    def distort(self, rays_external: np.ndarray) -> np.ndarray:
        "Apply distortion. Input/Output rays expressed in normalized coords (i.e. intersection with z=1)"

        raise NotImplementedError()

    def undistort(self, rays_internal: np.ndarray) -> np.ndarray:
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
    
        self.undistort_cache: dict[
            tuple[float, float, float],
            tuple[float, float, float]
        ] = {}

    def distort(self, rays_external: np.ndarray) -> np.ndarray:
        x = rays_external[..., 0]
        y = rays_external[..., 1]
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3

        # Radial distortion
        radial = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6

        # Tangential distortion
        x_tang = 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
        y_tang = self.p1*(r2 + 2*y**2) + 2*self.p2*x*y

        x_distorted = x*radial + x_tang
        y_distorted = y*radial + y_tang

        return np.stack([x_distorted, y_distorted], axis=-1)

    def undistort(self, rays_internal: np.ndarray) -> np.ndarray:
        def func(uv, uv_d):
            return self.distort(uv) - uv_d

        undistorted = []
        for d_pt in tqdm(rays_internal, desc="undistorting rays"):
            # attempt to find in the cache first
            lookup_key = tuple(d_pt)
            cache_lookup = self.undistort_cache.get(lookup_key, None)
            if cache_lookup is not None:
                undistorted.append(cache_lookup)
            else:
                # compute inverse numerically (expensive)
                # TODO: use second order solver
                solution = scipy.optimize.root(func, x0=d_pt, args=(d_pt,), method="hybr", tol=1e-3)
                val = solution.x
                self.undistort_cache[lookup_key] = val
                undistorted.append(val)

        return np.array(undistorted)
