import numpy as np
import open3d

from ..primitives import Pose3D
from ..camera_model import CameraModel


def simulate_capture(scene: list[open3d.geometry.TriangleMesh], camera: CameraModel, camera_pose: Pose3D) -> np.ndarray:
    """
    Perform a raycast image capture simulation of the given camera at the given position within a scene.
    Scene consists of colored meshes.

    One ray per pixel, from it's center.
    """

    # prepare rays
    W, H = camera.res_xy
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixel_centers = np.stack((x_coords, y_coords), axis=-1, dtype=np.float32)     # (H, W, 2)
    pixel_centers += 0.5                            # pixel center convention
    pixel_centers = pixel_centers.reshape((-1, 2))  # flatten

    ray_directions_sensor = camera.cast_ray_from_pixel(pixel_centers)
    ray_directions_world  = camera_pose.R.apply(ray_directions_sensor)

    N = W*H
    rays = np.zeros((N, 6), dtype=np.float32)
    rays[:,:3] = camera_pose.t
    rays[:,3:] = ray_directions_world

    # verify scene geoms all have color
    for geom in scene:
        assert geom.has_vertex_colors()

    # prepare the tensorized scene
    raycasting_scene = open3d.t.geometry.RaycastingScene()
    for geom in scene:
        geom_new = open3d.t.geometry.TriangleMesh.from_legacy(geom)
        raycasting_scene.add_triangles(geom_new)
    
    # perform the raycasting
    results = raycasting_scene.cast_rays(rays)

    # grab results
    hit = results["t_hit"].numpy().reshape(-1) < np.inf
    geom_hit_ids = results["geometry_ids"].numpy().reshape(-1)
    triangle_ids = results["primitive_ids"].numpy().reshape(-1)
    uvs = results["primitive_uvs"].numpy().reshape(-1, 2)  # shape (N,2)

    colors = np.full((N,3), fill_value=np.nan)

    # lookup color of each ray hit point
    for i, geom in enumerate(scene):
        mask = hit & (geom_hit_ids == i)
        if not np.any(mask):
            continue

        tris = np.asarray(geom.triangles)
        verts = np.asarray(geom.vertices)
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

    # reshape to image dims
    colors = colors.reshape((H, W, 3))
    return colors


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
