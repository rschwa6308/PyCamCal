import numpy as np
import cv2

from ...primitives.pose import Pose3D


class CheckerboardTarget:
    def __init__(self, nrows: int, ncols: int, square_size: float):
        self.nrows = nrows
        self.ncols = ncols
        self.square_size = square_size

    def get_feature_id(self, row: int, col: int):
        return f"CB-{row}-{col}"

    def detect(self, image: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """
        Attempt to detect a checkerboard calibration target within the given image.
        
        Returns sub-pixel coordinates of all detected calibration points: the inner cell corners (of which there are `(nrows-1)*(ncols-1)`).
        Each returned point is referenced by its (x, y) location within the grid of inner corners, indexed from the top-left.
        """

        # WARNING: 100% AI generated

        image_8bit = (image * 255).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # OpenCV uses inner corners: (ncols-1, nrows-1)
        pattern_size = (self.ncols - 1, self.nrows - 1)
        
        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
        if not ret:
            print("NO CHECKERBOARD DETECTED")
            return False, {}  # checkerboard not found

        # Refine corners to sub-pixel accuracy
        corners = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # remove extra dimension
        corners = corners.squeeze()

        # reverse ordering (want: tl -> br)
        corners = corners[::-1]

        # Map to (row, col) grid
        corner_dict = {}
        for i in range(pattern_size[1]):
            for j in range(pattern_size[0]):
                idx = i * pattern_size[0] + j
                key = self.get_feature_id(i, j)
                corner_dict[key] = corners[idx]
        
        return True, corner_dict

    def get_object_points(self, target_pose=Pose3D.identity()):
        """
        Generate the world-space points associated with this checkerboard calibration target.

        Target is defined as lying in the (+x, +y) quadrant of the z=0 plane, with row #0 at the top.
        """

        feature_points = {}

        # inner corners
        for row in range(self.nrows-1):
            for col in range(self.ncols-1):
                key = self.get_feature_id(row, col)
                pos_local = (
                    (col+1) * self.square_size,
                    (self.nrows-(row+1)) * self.square_size,
                    0.0
                )
                pos_world = target_pose.apply(pos_local)
                feature_points[key] = pos_world

        return feature_points
