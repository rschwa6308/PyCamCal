import numpy as np
from scipy.spatial.transform import Rotation as R3D


def get_lookat_orientation(cam_pos, target, up) -> R3D:
    """
    Get the rotation that orients a camera at `cam_pos` to look at `target`.
    Roll about the boresight axis is chosen to best match the given `up` vector.

    Camera frame convention: +x=right, +y=down, +z=forward
    """

    down_requested = -np.array(up)

    # boresight direction
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)

    # right direction (from requested up)
    right = np.cross(down_requested, fwd)

    # actual down (orthonormal)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)

    R_cam2world = np.column_stack((right, down, fwd))
    return R3D.from_matrix(R_cam2world)


def get_random_rotation(magnitude: float, degrees=False) -> R3D:
    # random axis (uniform from suface of a sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    return R3D.from_rotvec(axis * magnitude, degrees=degrees)
