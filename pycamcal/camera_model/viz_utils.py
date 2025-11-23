import numpy as np
from matplotlib import pyplot as plt

from .camera_model import CameraModel

def visualize_distortion_model(camera: CameraModel, subsample=50, exaggeration=1.0, ax=None):
    "Visualize a camera's distortion model as a 2D plot showing the warp vector field per-pixel"

    W, H = camera.res_xy
    y_coords, x_coords = np.meshgrid(
        np.arange(H)[::subsample],
        np.arange(W)[::subsample],
        indexing="ij"
    )

    pixel_centers = np.stack((x_coords, y_coords), axis=-1, dtype=np.float32)     # (H, W, 2)
    pixel_centers += 0.5                            # pixel center convention
    pixel_centers = pixel_centers.reshape((-1, 2))  # flatten

    ray_directions_internal = camera.cast_ray_from_pixel(pixel_centers, include_distortion=False)
    ray_directions_external = camera.cast_ray_from_pixel(pixel_centers, include_distortion=True)

    # intersection of distorted rays with z=1 plane
    points_internal = ray_directions_internal[:,:2] / ray_directions_internal[:,2:3]
    points_external = ray_directions_external[:,:2] / ray_directions_external[:,2:3]

    point_deltas = points_external - points_internal

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.quiver(
        points_internal[:,0], points_internal[:,1],
        point_deltas[:,0], point_deltas[:,1],
        scale=1.0/exaggeration
    )

    return ax
