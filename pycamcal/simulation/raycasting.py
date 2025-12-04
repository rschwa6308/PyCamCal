import numpy as np
import open3d

from ..primitives import Pose3D
from ..camera_model import CameraModel
from ..primitives.math_helpers import is_perfect_square


def perform_raycast(scene: list[open3d.geometry.TriangleMesh], ray_origins: np.ndarray, ray_directions: np.ndarray):
    N = max(len(ray_origins), len(ray_directions))

    rays = np.zeros((N, 6), dtype=np.float32)
    rays[:,:3] = ray_origins
    rays[:,3:] = ray_directions

    # prepare the tensorized scene
    raycasting_scene = open3d.t.geometry.RaycastingScene()
    for geom in scene:
        geom_new = open3d.t.geometry.TriangleMesh.from_legacy(geom)
        raycasting_scene.add_triangles(geom_new)

    # perform the raycasting
    results = raycasting_scene.cast_rays(rays)
    return results


def get_subpixel_uniform_sampling_pattern(s: int) -> np.ndarray:
    """
    Subdide pixel into `s` x `s` sub-regions. Sample one point from the center of each sub-region.
    
    Return array of shape `(s, s, 2)` floating point sub-pixel coordinates in [0.0, 1.0]^2
    """

    pattern = np.zeros((s, s, 2), dtype=np.float32)

    step = 1.0 / s
    offset = step / 2.0
    for i in range(s):
        for j in range(s):
            pattern[i, j, 0] = i * step + offset
            pattern[i, j, 1] = j * step + offset

    return pattern


def simulate_capture(scene: list[open3d.geometry.TriangleMesh], camera: CameraModel, camera_pose: Pose3D, rays_per_pixel: int = 1) -> np.ndarray:
    """
    Perform a raycast image capture simulation of the given camera at the given position within a scene.
    Scene consists of colored meshes.

    By default, casts one ray per pixel (from it's center).
    """

    assert is_perfect_square(rays_per_pixel)

    # verify scene geoms all have color
    for geom in scene:
        assert geom.has_vertex_colors()

    W, H = camera.res_xy

    # array of pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixel_tl_corners = np.stack((x_coords, y_coords), axis=-1, dtype=np.float32)            # (H, W, 2)
    pixel_tl_corners = pixel_tl_corners.reshape(-1, 2)                                      # (H*W, 2)

    # sub-pixel sampling pattern
    s = int(np.sqrt(rays_per_pixel))
    subpixel_pattern = get_subpixel_uniform_sampling_pattern(s)                             # (s, s, 2)
    subpixel_pattern = subpixel_pattern.reshape(-1, 2)                                      # (s*s, 2)

    subpixel_ray_sources = pixel_tl_corners[:, None, :] + subpixel_pattern[None, :, :]      # (H*W, s*s, 2)
    subpixel_ray_sources = subpixel_ray_sources.reshape(-1, 2)                              # (H*W*s*s, 2)

    ray_directions_sensor = camera.cast_ray_from_pixel(subpixel_ray_sources)                # (H*W*s*s, 3)
    ray_directions_world  = camera_pose.R.apply(ray_directions_sensor)                      # (H*W*s*s, 3)

    # compute ray-scene intersections
    results = perform_raycast(scene, ray_origins=camera_pose.t, ray_directions=ray_directions_world)

    # parse results
    hit = results["t_hit"].numpy().reshape(-1) < np.inf
    geom_hit_ids = results["geometry_ids"].numpy().reshape(-1)
    triangle_ids = results["primitive_ids"].numpy().reshape(-1)
    uvs = results["primitive_uvs"].numpy().reshape(-1, 2)  # shape (N,2)

    colors = np.full((len(subpixel_ray_sources), 3), dtype=np.float32, fill_value=np.nan)

    # lookup color of each ray hit point
    for i, geom in enumerate(scene):
        mask = hit & (geom_hit_ids == i)
        if not np.any(mask):
            continue

        tris = np.asarray(geom.triangles)
        vcolors = np.asarray(geom.vertex_colors)

        prim_ids_hit = triangle_ids[mask]
        uv = uvs[mask]

        # Compute barycentric weights
        u_vals = uv[:, 0]
        v_vals = uv[:, 1]
        w_vals = 1.0 - u_vals - v_vals

        # Triangle vertex indices
        tri_indices = tris[prim_ids_hit]  # shape (num_hits, 3)

        # Vertex colors
        c0 = vcolors[tri_indices[:, 0]]
        c1 = vcolors[tri_indices[:, 1]]
        c2 = vcolors[tri_indices[:, 2]]

        # Interpolate
        colors[mask] = w_vals[:, None] * c0 + u_vals[:, None] * c1 + v_vals[:, None] * c2

    # clip colors to [0, 1] (for numerical stability)
    colors = np.clip(colors, 0.0, 1.0)

    # average color per-pixel
    colors = colors.reshape(H, W, s*s, 3)
    colors_avg = np.mean(colors, axis=2)        # TODO: better color-space averaging

    return colors_avg

    # return np.rollaxis(colors.reshape(H, W, s, s, 3), 2, 1).reshape(H*s, W*s, 3)


if __name__ == "__main__":
    # Create meshes and convert to open3d.t.geometry.TriangleMesh
    cube = open3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
    torus = open3d.geometry.TriangleMesh.create_torus().translate([0, 0, 2])
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate(
        [1, 2, 3])
    
    cube.paint_uniform_color([1, 0, 0])
    torus.paint_uniform_color([0, 1, 0])
    sphere.paint_uniform_color([0, 0, 1])

    rays = open3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[0, 0, 2],
        eye=[2, 3, 0],
        up=[0, 1, 0],
        width_px=1000,
        height_px=1000,
    )

    res = simulate_capture([cube, torus, sphere], rays)
    # print(res)
    print(res.shape)

    from matplotlib import pyplot as plt

    plt.imshow(res)
    plt.show()
