import pprint
from typing import Literal
import jax.numpy as jnp

from ..optimization.optimization_quantity import VALUE, Unknown

from .distortion_model import DistortionModel


class CameraModel:
    def __init__(self, res_xy: tuple[int, int], fx, fy, cx, cy, distortion: DistortionModel):
        self.res_xy = jnp.array(res_xy, dtype=int)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.distortion = distortion

    @staticmethod
    def from_fov(res_xy, fov_xy, distortion: DistortionModel=None, degrees=True) -> "CameraModel":
        if degrees:
            fov_x, fov_y = jnp.deg2rad(jnp.array(fov_xy))
        else:
            fov_x, fov_y = fov_xy

        width, height = res_xy
        fx = (width / 2) / jnp.tan(fov_x / 2)
        fy = (height / 2) / jnp.tan(fov_y / 2)
        cx = width / 2
        cy = height / 2

        return CameraModel(res_xy, fx, fy, cx, cy, distortion)

    def get_intrinsics_matrix(self):
        return jnp.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,      0.0,    1.0    ]
        ])
    
    def get_focals(self):
        return jnp.array([VALUE(self.fx), VALUE(self.fy)])
    
    def get_centers(self):
        return jnp.array([VALUE(self.cx), VALUE(self.cy)])

    def get_fov(self, degrees=False) -> tuple[float, float]:
        width, height = self.res_xy
        fov_x = 2 * jnp.arctan((width / 2) / self.fx)
        fov_y = 2 * jnp.arctan((height / 2) / self.fy)

        if degrees:
            fov_x = jnp.rad2deg(fov_x)
            fov_y = jnp.rad2deg(fov_y)

        return fov_x, fov_y
    
    def project_into_image(self, points: jnp.ndarray, include_distortion=True):
        "Project world-space point(s) into the image frame, returning sub-pixel sensor intersection coordinates"

        # normalize points by intersecting incoming rays with z=1 plane
        points_external = points[...,:2] / points[...,2:3]

        # apply lens distortion
        if self.distortion is not None and include_distortion:
            points_internal = self.distortion.distort(points_external)
        else:
            points_internal = points_external

        # apply pinhole projection
        pixel_coords = points_internal * self.get_focals() + self.get_centers()

        return pixel_coords

    def cast_ray_from_pixel(self, pixel_coords: jnp.ndarray, normalized=True, include_distortion=True):
        "Cast ray(s) from the given (sub)pixel coordinate(s)"

        # invert pinhole projection
        points_internal = (pixel_coords - self.get_centers()) / self.get_focals()

        # invert lens distortion
        if self.distortion is not None and include_distortion:
            points_external = self.distortion.undistort(points_internal)
        else:
            points_external = points_internal

        rays = jnp.hstack([points_external, jnp.ones((len(points_external), 1))])

        if normalized:
            rays /= jnp.linalg.norm(rays, axis=1, keepdims=True)

        return rays

    def collect_unknowns(self) -> list[Unknown]:
        # intrinsics params
        unknowns = [
            param for param in (self.fx, self.fy, self.cx, self.cy)
            if isinstance(param, Unknown)
        ]

        # distortion params
        unknowns.extend(self.distortion.collect_unknowns())

        return unknowns
    
    def to_dict(self) -> dict:
        return {
            "res_xy": self.res_xy.tolist(),
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "distortion": self.distortion.to_dict()
        }

    def __str__(self):
        return pprint.pformat(self.to_dict(), sort_dicts=False)
