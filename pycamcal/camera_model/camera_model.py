from typing import Literal
import numpy as np

from .distortion_model import DistortionModel


class CameraModel:
    def __init__(self, res_xy: tuple[int, int], fx, fy, cx, cy, distortion: DistortionModel):
        self.res_xy = np.array(res_xy, dtype=int)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.distortion = distortion
    
    @staticmethod
    def from_fov(res_xy, fov_xy, distortion: DistortionModel=None, degrees=True) -> "CameraModel":
        if degrees:
            fov_x, fov_y = np.deg2rad(fov_xy[0]), np.deg2rad(fov_xy[1])
        else:
            fov_x, fov_y = fov_xy

        width, height = res_xy
        fx = (width / 2) / np.tan(fov_x / 2)
        fy = (height / 2) / np.tan(fov_y / 2)
        cx = width / 2
        cy = height / 2

        return CameraModel(res_xy, fx, fy, cx, cy, distortion)

    def get_instrinsics_matrix(self):
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,      0.0,    1.0    ]
        ])
    
    def get_fov(self, degrees=False) -> tuple[float, float]:
        width, height = self.res_xy
        fov_x = 2 * np.arctan((width / 2) / self.fx)
        fov_y = 2 * np.arctan((height / 2) / self.fy)

        if degrees:
            fov_x = np.rad2deg(fov_x)
            fov_y = np.rad2deg(fov_y)

        return fov_x, fov_y

    def cast_ray_from_pixel(self, pixel_coords: np.ndarray, normalized=True, include_distortion=True):
        "Cast ray(s) from the given (sub)pixel coordinate(s)"

        K = self.get_instrinsics_matrix()
        K_inv = np.linalg.inv(K)

        # construct homogenous vectors
        pixel_coords_homog = np.hstack([pixel_coords, np.ones((len(pixel_coords), 1))])

        # invert pinhole projection
        points_internal = (K_inv @ pixel_coords_homog.T).T

        # intersect with z=1 plane
        points_internal = points_internal[:,:2] / points_internal[:,2:3]

        # invert lens distortion
        if self.distortion is not None and include_distortion:
            points_external = self.distortion.undistort(points_internal)
        else:
            points_external = points_internal

        rays = np.hstack([points_external, np.ones((len(points_external), 1))])

        if normalized:
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        return rays
