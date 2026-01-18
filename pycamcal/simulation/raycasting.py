import numpy as np
import open3d

from ..primitives import Pose3D
from ..camera_model import CameraModel
from ..primitives.math_helpers import is_perfect_square

from .materials import *

# triangle "color" value
MIRROR_COLOR_CODE = [1.0, 0.0, 1.0]     # magenta


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


def reflect_off_surface(d, n):
    """
    Reflect ray(s) with direction `d` off surface with normal(s) `n`.
    
    Both d and n can be shape (..., 3) and broadcasting is supported.
    Returns reflected vectors of same shape.
    """
    return d - 2 * np.sum(d * n, axis=-1, keepdims=True) * n



def round(scene, ray_origins, ray_directions, use_triangle_material_ids=True):

    n = len(ray_origins)
    ray_colors      = np.full((n, 3), dtype=np.float32, fill_value=np.nan)
    terminated_mask = np.full((n,  ), dtype=bool,       fill_value=False)

    # compute ray-scene intersections
    results = perform_raycast(scene, ray_origins, ray_directions)

    # parse results
    hit_range       = results["t_hit"].numpy().reshape(-1)                  # (N,)
    geom_hit_ids    = results["geometry_ids"].numpy().reshape(-1)           # (N,)
    tri_hit_ids     = results["primitive_ids"].numpy().reshape(-1)          # (N,)
    tri_hit_normals = results["primitive_normals"].numpy().reshape(-1, 3)   # (N, 3)
    uvs             = results["primitive_uvs"].numpy().reshape(-1, 2)       # (N, 2)

    hit_mask = hit_range < np.inf
    hit_points = ray_origins + ray_directions * hit_range[:,None]

    terminated_mask[~hit_mask] = True

    # lookup color of each ray hit point
    for i, geom in enumerate(scene):
        mask = hit_mask & (geom_hit_ids == i)
        if not np.any(mask):
            continue

        mask_where = np.flatnonzero(mask)

        tris = np.asarray(geom.triangles)
        vcolors = np.asarray(geom.vertex_colors)
        tri_mats = np.asarray(geom.triangle_material_ids)

        tri_ids = tri_hit_ids[mask]

        if use_triangle_material_ids:
            mat_ids = tri_mats[tri_ids]

            mask_mirror      = (mat_ids == MAT_MIRROR)
            mask_transparent = (mat_ids == MAT_TRANSPARENT)
            mask_terminal    = ~(mask_mirror | mask_transparent)

            if np.any(mask_mirror):      print(f"Num mirror hits:      {np.count_nonzero(mask_mirror)}")
            if np.any(mask_transparent): print(f"Num transparent hits: {np.count_nonzero(mask_transparent)}")

            # handle simple (terminal/absorptive) materials
            where_terminal = mask_where[mask_terminal]
            ray_colors[where_terminal] = lookup_material_color(mat_ids[mask_terminal])
            terminated_mask[where_terminal] = True

            # handle transparent materials
            pass    # TODO
            where_transparent = mask_where[mask_transparent]
            if np.any(mask_transparent): print(ray_origins[where_transparent][0])

            # handle mirrors
            where_mirror = mask_where[mask_mirror]
            ray_directions[where_mirror] = reflect_off_surface(ray_directions[where_mirror], tri_hit_normals[where_mirror])

        else:
            raise NotImplementedError()

    live_mask = ~terminated_mask
    remaining_live_rays = [hit_points[live_mask], ray_directions[live_mask]]

    return terminated_mask, ray_colors[terminated_mask], remaining_live_rays



def simulate_capture(scene: list[open3d.geometry.TriangleMesh], camera: CameraModel, camera_pose: Pose3D, rays_per_pixel: int = 1, use_triangle_material_ids=True, use_vertex_colors=False) -> np.ndarray:
    """
    Perform a raycast image capture simulation of the given camera at the given position within a scene.
    Scene consists of colored meshes.

    By default, casts one ray per pixel (from it's center).
    """

    assert is_perfect_square(rays_per_pixel)
    assert not (use_triangle_material_ids and use_vertex_colors)

    # verify all scene geoms have necessary appearance information
    for geom in scene:
        if use_triangle_material_ids:
            assert geom.has_triangle_material_ids()
        if use_vertex_colors:
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

    N = len(subpixel_ray_sources)                                                           # N = H*W*s*s

    ray_origins_world = np.full((N, 3), fill_value=camera_pose.t)                           # (N, 3)

    ray_directions_sensor = camera.cast_ray_from_pixel(subpixel_ray_sources)                # (N, 3)
    ray_directions_world  = camera_pose.R.apply(ray_directions_sensor)                      # (N, 3)
    ray_directions_world = np.array(ray_directions_world)

    # raycasting output products
    final_ray_colors    = np.full((N, 3), dtype=np.float32, fill_value=np.nan)
    rays_finalized_mask = np.full((N,  ), dtype=bool,       fill_value=False)

    # iterate: initial raycast, first bounce, second bounce, ...
    live_rays = [ray_origins_world, ray_directions_world]
    round_count = 0
    while not np.all(rays_finalized_mask):
        print(f"Bounce depth: {round_count} | Rays in flight: {len(live_rays[0])}")
        live_indices = np.where(~rays_finalized_mask)[0]

        if round_count > 10: break

        terminated_mask, ray_colors, remaining_live_rays = round(scene, *live_rays, use_triangle_material_ids)
        final_ray_colors[live_indices[terminated_mask]] = ray_colors
        rays_finalized_mask[live_indices[terminated_mask]] = True

        live_rays = remaining_live_rays

        # advance rays a very small amount before next round to avoid intersecting the same surface again
        live_rays[0] += 1e-4 * live_rays[1]

        round_count += 1
        print()

    # clip colors to [0, 1] (for numerical stability)
    final_ray_colors = np.clip(final_ray_colors, 0.0, 1.0)

    # average color per-pixel
    final_ray_colors = final_ray_colors.reshape(H, W, s*s, 3)
    colors_avg = np.mean(final_ray_colors, axis=2)        # TODO: better color-space averaging

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
