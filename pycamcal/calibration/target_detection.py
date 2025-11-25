import numpy as np
import cv2


def detect_checkerboard_target(image: np.ndarray, nrows: int, ncols: int, color_a=[0,0,0], color_b=[1,1,1]) -> dict[tuple[int, int], np.ndarray]:
    """
    Attempt to detect a checkerboard calibration target within the given image.
    
    Returns sub-pixel coordinates of all detected calibration points: the inner cell corners (of which there are `(nrows-1)*(ncols-1)`).
    Each returned point is referenced by its (x, y) location within the grid of inner corners, indexed from the top-left.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # OpenCV uses inner corners: (ncols-1, nrows-1)
    pattern_size = (ncols - 1, nrows - 1)
    
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return {}  # Checkerboard not found

    # Refine corners to sub-pixel accuracy
    corners = cv2.cornerSubPix(
        gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Map to (row, col) grid
    corner_dict = {}
    for i in range(pattern_size[1]):  # rows
        for j in range(pattern_size[0]):  # cols
            idx = i * pattern_size[0] + j
            corner_dict[(i, j)] = corners[idx, 0]  # corners[i,j] is (x, y)
    
    return corner_dict
