import numpy as np
from scipy.spatial.transform import Rotation as R3D
from tqdm import tqdm

from ..primitives.pose import Pose3D
from ..primitives.math_helpers import get_lookat_orientation, get_random_rotation
from ..camera_model.camera_model import CameraModel

from .cornell_box import create_cornell_box
from .checkerboard import create_checkerboard_mesh
from .raycasting import simulate_capture


def generate_calibration_dataset(camera_model: CameraModel, num_images: int,  random_seed: float = None) -> tuple[list[np.ndarray], list[Pose3D], dict, list]:
    """
    Generate a synthetic camera calibration dataset consisting of a checkerboard calibration target placed in a cornell box, captured from a random sampling of different camera poses.

    Returns a list of images along with their associated ground-truth camera pose.
    Also returns calibration target info and scene contents for future reference.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    ##################################################
    # Prepare the scene
    ##################################################

    # create basic environment for some nice visual context
    room_width = 6
    room_depth = 6
    room_height = 5
    cornell_box = create_cornell_box(width=room_width, height=room_height, depth=room_depth)

    # create calibration target
    checkerboard, target_points = create_checkerboard_mesh(num_rows=7, num_cols=10, square_size=0.25)
    target_pose = Pose3D(
        -checkerboard.get_center() + [0, 3, 1],
        R3D.from_euler("x", 90, degrees=True)
    )
    checkerboard.transform(target_pose.as_transformation_matrix())
    target_points = target_pose.apply(target_points)

    scene = [*cornell_box.values(), checkerboard]

    ##################################################
    # Sample camera poses
    ##################################################

    target_center = checkerboard.get_center()

    # random positions from the front half of the room (with some margin away from the walls)
    margin = 0.2
    camera_positions = np.random.uniform(
        low  = np.array([-room_width/2, -room_depth/2, 0.0        ]) + margin,
        high = np.array([+room_width/2,  0.0,          room_height]) - margin,
        size=(num_images, 3)
    )

    # orient each camera to lookat the target center, and then perterb by a random amount
    camera_orientations = []
    for cam_pos in camera_positions:
        # aim at calibration target
        cam_ori = get_lookat_orientation(cam_pos, target_center, up=[0,0,1])

        # small random pan/tilt
        fov_x, fov_y = camera_model.get_fov()
        s = 0.25
        pan  = np.random.uniform(-fov_x*s, +fov_x*s)
        tilt = np.random.uniform(-fov_y*s, +fov_y*s)
        perturbation = R3D.from_euler("XY", [pan, tilt])
        cam_ori *= perturbation

        camera_orientations.append(cam_ori)

    camera_poses = [Pose3D(pos, ori) for pos, ori in zip(camera_positions, camera_orientations)]

    ##################################################
    # Simulate image captures
    ##################################################

    images = []

    for i, camera_pose in tqdm(enumerate(camera_poses), total=len(camera_poses), desc="Simulating camera captures"):
        image = simulate_capture(
            scene,
            camera_model,
            camera_pose,
            rays_per_pixel=9    # anti-aliasing
        )
        images.append(image)

    target_info = {
        "target_type": "checkerboard",
        "params": {
            "nrows": 7,
            "ncols": 10,
            "square_size": 0.25
        },
        "pose": {
            "position": list(target_pose.t),
            "orientation": list(target_pose.R.as_quat())
        }
    }
    
    return images, camera_poses, target_info, scene
