import numpy as np
import open3d

from ..primitives.pose import Pose3D
from ..camera_model.camera_model import CameraModel


def get_frustum_wireframe(fov_x: float, fov_y: float, scale: float = 1.0) -> open3d.geometry.LineSet:
    "Get a simple wireframe visualization of the view-frustum of a sensor with the given field of view"

    # Compute half-dimensions on the z=scale plane
    hx = np.tan(fov_x / 2) * scale
    hy = np.tan(fov_y / 2) * scale

    # 5 key points: apex + 4 corners of near plane
    pts = np.array([
        [0,   0,   0],        # 0: apex
        [-hx, -hy, scale],    # 1
        [ hx, -hy, scale],    # 2
        [ hx,  hy, scale],    # 3
        [-hx,  hy, scale],    # 4
    ])

    # Lines: apex->corners and edges of near plane
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]


    ls = open3d.geometry.LineSet()
    ls.points = open3d.utility.Vector3dVector(pts)
    ls.lines = open3d.utility.Vector2iVector(lines)
    return ls


def visualize_camera_positioning(scene: list[open3d.geometry.TriangleMesh], camera: CameraModel, camera_poses: list[Pose3D], camera_viz_scale=1.0):
    "Visualize a list of cameras positioned/oriented within a scene"

    fov_x, fov_y = camera.get_fov()

    cam_viz_geoms = []
    for cam_pose in camera_poses:
        axes = open3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_viz_scale)
        frustum = get_frustum_wireframe(fov_x, fov_y, scale=camera_viz_scale)

        axes.transform(cam_pose.as_transformation_matrix())
        frustum.transform(cam_pose.as_transformation_matrix())

        cam_viz_geoms.append(axes)
        cam_viz_geoms.append(frustum)

    scene_axes = open3d.geometry.TriangleMesh.create_coordinate_frame()

    open3d.visualization.draw_geometries([*scene, scene_axes, *cam_viz_geoms], mesh_show_back_face=True)
