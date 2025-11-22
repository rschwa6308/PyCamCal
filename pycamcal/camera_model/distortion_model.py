from abc import ABC
from dataclasses import dataclass

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


@dataclass
class RadialTangentialDistortion(DistortionModel):
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

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
            solution = scipy.optimize.root(func, x0=d_pt, args=(d_pt,), method="hybr")
            undistorted.append(solution.x)

        return np.array(undistorted)
