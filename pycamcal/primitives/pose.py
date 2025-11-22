from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R3D

@dataclass
class Pose3D:
    R: R3D
    t: np.ndarray
