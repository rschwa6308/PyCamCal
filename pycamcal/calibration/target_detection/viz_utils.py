

import numpy as np
from matplotlib import pyplot as plt


def plot_target_detection(image: np.ndarray, target_points: dict[str, np.ndarray], ax=None):
    "Plot the given image annotated with the given target points"

    points = np.array([
        target_points[key] for key in sorted(target_points.keys())
    ])

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image)
    ax.scatter(points[:,0], points[:,1], color="magenta", marker="x", label="target detections")

    ax.legend()

    return ax
