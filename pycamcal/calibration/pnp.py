from matplotlib import pyplot as plt
import numpy as np

from ..primitives.pose import Pose3D, R3D
from ..primitives.rigid_body_alignment import estimate_rigid_body_alignment
from ..camera_model.camera_model import CameraModel


def solve_P3P(world_points: np.ndarray, image_points: np.ndarray, camera_model: CameraModel) -> Pose3D:
    "Special case of the Perspective-n-Point problem where n=3"

    assert(len(world_points) == len(image_points) == 3)

    bearings = camera_model.cast_ray_from_pixel(image_points, normalized=True)
    f1, f2, f3 = bearings

    assert(np.isclose(1.0, np.linalg.norm(f1)))
    assert(np.isclose(1.0, np.linalg.norm(f2)))
    assert(np.isclose(1.0, np.linalg.norm(f3)))

    # triangle side lengths
    P1, P2, P3 = world_points
    s12 = np.linalg.norm(P1 - P2)
    s13 = np.linalg.norm(P1 - P3)
    s23 = np.linalg.norm(P2 - P3)

    # angles between bearing vectors
    m12 = np.dot(f1, f2)
    m13 = np.dot(f1, f3)
    m23 = np.dot(f2, f3)

    # Grunert quartic polynomial
    A4 = -4*m23**2*s12**2*s13**2 + s12**4 + 2*s12**2*s13**2 - 2*s12**2*s23**2 + s13**4 - 2*s13**2*s23**2 + s23**4
    A3 = 8*m12*m23**2*s12**2*s13**2 - 4*m12*s12**2*s13**2 + 4*m12*s12**2*s23**2 - 4*m12*s13**4 + 8*m12*s13**2*s23**2 - 4*m12*s23**4 - 4*m13*m23*s12**4 + 4*m13*m23*s12**2*s13**2 + 4*m13*m23*s12**2*s23**2
    A2 = 4*m12**2*s13**4 - 8*m12**2*s13**2*s23**2 + 4*m12**2*s23**4 - 8*m12*m13*m23*s12**2*s13**2 - 8*m12*m13*m23*s12**2*s23**2 + 4*m13**2*s12**4 - 4*m13**2*s12**2*s23**2 + 4*m23**2*s12**4 - 4*m23**2*s12**2*s13**2 - 2*s12**4 + 2*s13**4 - 4*s13**2*s23**2 + 2*s23**4
    A1 = 8*m12*m13**2*s12**2*s23**2 + 4*m12*s12**2*s13**2 - 4*m12*s12**2*s23**2 - 4*m12*s13**4 + 8*m12*s13**2*s23**2 - 4*m12*s23**4 - 4*m13*m23*s12**4 + 4*m13*m23*s12**2*s13**2 + 4*m13*m23*s12**2*s23**2
    A0 = -4*m13**2*s12**2*s23**2 + s12**4 - 2*s12**2*s13**2 + 2*s12**2*s23**2 + s13**4 - 2*s13**2*s23**2 + s23**4

    roots = np.roots([A4, A3, A2, A1, A0])
    print("roots:", roots)

    # recover poses (up to 4 possible numerical solutions, only one actually valid)
    candidate_poses = []

    for u in roots:
        if np.imag(u) > 1e-8:
            continue

        u = np.real(u)

        if u < 1e-12:
            continue

        v =  m13 + np.sqrt(-2*m12*s13**2*u + m13**2*s12**2 - s12**2 + s13**2*u**2 + s13**2)/s12
        if not np.isfinite(v) or v <= 0:
            continue

        # ranges
        d1 = s13 / np.sqrt(-2*m13*v + v**2 + 1)
        d2 = u * d1
        d3 = v * d1

        print("ranges:", d1, d2, d3)

        Xc = np.column_stack([d1*f1, d2*f2, d3*f3])
        Xw = world_points.T

        R, t = estimate_rigid_body_alignment(Xc, Xw)
        pose = Pose3D(t, R3D.from_matrix(R))
        candidate_poses.append(pose)

    return candidate_poses

    # # find the valid solution via reprojection error
    # best_pose = None
    # best_error = +np.inf

    # for pose in candidate_poses:
    #     points_external = pose.inv().apply(world_points)
    #     points_reproj = camera_model.project_into_image(points_external)

    #     plt.scatter(*points_reproj.T)
    #     plt.xlim(0, camera_model.res_xy[0])
    #     plt.ylim(camera_model.res_xy[1], 0)

    #     error = np.sum(np.linalg.norm(points_reproj - image_points, axis=-1))
    #     print("error:", error)

    #     if error < best_error:
    #         best_error = error
    #         best_pose = pose

    # return best_pose



def solve_PnP(world_points: np.ndarray, image_points: np.ndarray, camera_model: CameraModel) -> Pose3D:
    """
    Solve the Perspective-n-Point problem by estimating the 6-DOF pose of a camera given a set of 3D-2D point correspondences.

     - `world_points`: 3D location of the known points in world-space
     - `image_points`: 2D sub-pixel location of corresponding detected points in image-space
     - `camera_model`: pinhole intrinsics + distortion model of the camera
    """

