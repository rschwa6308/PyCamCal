from abc import ABC
from dataclasses import dataclass
from functools import lru_cache

import scipy
import jax.scipy.optimize
import numpy as np
import jax.numpy as jnp
import scipy.optimize
from tqdm import tqdm

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

        self.undistort_cache: dict[
            tuple[float, float, float],
            tuple[float, float, float]
        ] = {}

        N = 4000

        # grid representing [-2, 2] x [-2, 2] @ 0.001/px (quantized)
        self.undistort_cache = np.full((N, N, 2), np.nan, dtype=np.float32)

    def _make_cache_key(self, point: np.ndarray):
        return ((point + 2.0) * 1000).astype(int)

    def distort(self, rays_external: jnp.ndarray) -> jnp.ndarray:
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

        return jnp.stack([x_distorted, y_distorted], axis=-1)

    def undistort(self, rays_internal: jnp.ndarray) -> jnp.ndarray:
        result = np.zeros_like(rays_internal)
        cache_hit_mask = np.zeros(len(rays_internal), dtype=bool)

        # attempt cache lookup first
        keys = self._make_cache_key(rays_internal)
        cache_result = self.undistort_cache[keys[:,1], keys[:,0]]
        cache_hit_mask = np.all(np.isfinite(cache_result), axis=-1)
        result[cache_hit_mask] = cache_result[cache_hit_mask]

        # print(f"Undistort cache hit rate: {np.count_nonzero(cache_hit_mask) / len(cache_hit_mask):.0%}")

        if np.all(cache_hit_mask):
            return result

        # expensive numeric inverse
        numerical_result = self._undistort_numeric(rays_internal[~cache_hit_mask])
        result[~cache_hit_mask] = numerical_result

        # save new results to cache
        keys_miss = keys[~cache_hit_mask]
        self.undistort_cache[keys_miss[:,1], keys_miss[:,0]] = numerical_result

        return result
    
    def _undistort_numeric(self, rays_internal: jnp.ndarray) -> jnp.ndarray:

        # objective function
        def func_residual(uv, uv_d):
            reproj_error = self.distort(uv) - uv_d
            return reproj_error

        # jacobian
        func_jac = jax.vmap(jax.jacobian(func_residual))

        # jit for speed
        func_jit     = jax.jit(func_residual)
        func_jac_jit = jax.jit(func_jac)

        r2 = jnp.sum(rays_internal*rays_internal, axis=-1)
        r4 = r2**2
        r6 = r2**3
        radial = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        initial_guess = rays_internal * (1.0 / radial)[:,None]

        undistorted = gauss_newton_batched(
            func_jit,
            func_jac_jit,
            args=(rays_internal,),
            x0=initial_guess,
            max_iters=5,
            ftol=1e-3,
            damping=10.0,   # significant damping needed for stability in region r > 1
            verbose=True
        )

        return undistorted
