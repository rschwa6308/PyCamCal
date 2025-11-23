from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R3D

@dataclass
class Pose3D:
    t: np.ndarray
    R: R3D

    def as_transformation_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R.as_matrix()
        T[:3,  3] = self.t
        return T
